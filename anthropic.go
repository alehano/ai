package ai

import (
	"context"
	"errors"
	"fmt"
	"io"

	"github.com/liushuangls/go-anthropic/v2"
)

type AnthropicGen struct {
	client      *anthropic.Client
	model       string
	maxTokens   int
	temperature float32
	cachePrompt bool
}

func NewAnthropicGen(apiKey, model string, maxTokens int, temperature float32, cachePrompt bool) *AnthropicGen {
	client := anthropic.NewClient(apiKey)
	if cachePrompt {
		client = anthropic.NewClient(
			apiKey,
			anthropic.WithBetaVersion(anthropic.BetaPromptCaching20240731),
		)
	}

	return &AnthropicGen{
		client:      client,
		model:       model,
		maxTokens:   maxTokens,
		temperature: temperature,
		cachePrompt: cachePrompt,
	}
}

func (a *AnthropicGen) Generate(ctx context.Context, systemPrompt, prompt string) (string, error) {
	req := anthropic.MessagesRequest{
		Model:       anthropic.Model(a.model),
		Temperature: &a.temperature,
		MaxTokens:   a.maxTokens,
		Messages: []anthropic.Message{
			anthropic.NewUserTextMessage(prompt),
		},
	}

	if systemPrompt != "" {
		if a.cachePrompt {
			req.MultiSystem = []anthropic.MessageSystemPart{
				{
					Type: "text",
					Text: systemPrompt,
					CacheControl: &anthropic.MessageCacheControl{
						Type: anthropic.CacheControlTypeEphemeral,
					},
				},
			}
		} else {
			req.System = systemPrompt
		}
	}

	resp, err := a.client.CreateMessages(ctx, req)
	if err != nil {
		var apiErr *anthropic.APIError
		if errors.As(err, &apiErr) {
			return "", errors.New(apiErr.Message)
		}
		return "", err
	}

	return resp.Content[0].GetText(), nil
}

func (a *AnthropicGen) GenerateStream(ctx context.Context, systemPrompt, prompt string, resultCh chan string, doneCh chan bool, errCh chan error) {
	req := anthropic.MessagesStreamRequest{
		MessagesRequest: anthropic.MessagesRequest{
			Model:       anthropic.Model(a.model),
			Temperature: &a.temperature,
			MaxTokens:   a.maxTokens,
			Messages: []anthropic.Message{
				anthropic.NewUserTextMessage(prompt),
			},
		},
		OnContentBlockDelta: func(data anthropic.MessagesEventContentBlockDeltaData) {
			if data.Delta.Text != nil {
				select {
				case resultCh <- *data.Delta.Text:
				case <-ctx.Done():
					return
				}
			}
		},
		OnContentBlockStop: func(data anthropic.MessagesEventContentBlockStopData, content anthropic.MessageContent) {
			select {
			case doneCh <- true:
			case <-ctx.Done():
			}
		},
	}

	if systemPrompt != "" {
		if a.cachePrompt {
			req.MultiSystem = []anthropic.MessageSystemPart{
				{
					Type: "text",
					Text: systemPrompt,
					CacheControl: &anthropic.MessageCacheControl{
						Type: anthropic.CacheControlTypeEphemeral,
					},
				},
			}
		} else {
			req.System = systemPrompt
		}
	}

	_, err := a.client.CreateMessagesStream(ctx, req)
	if err != nil {
		if err == io.EOF {
			// Stream completed successfully
			select {
			case doneCh <- true:
			case <-ctx.Done():
			}
		} else {
			var apiErr *anthropic.APIError
			if errors.As(err, &apiErr) {
				select {
				case errCh <- errors.New(apiErr.Message):
				case <-ctx.Done():
				}
			} else {
				select {
				case errCh <- err:
				case <-ctx.Done():
				}
			}
		}
		return
	}

	// Wait for the context to be done
	<-ctx.Done()
}

func (a *AnthropicGen) GetModel() string {
	return a.model
}

func (a *AnthropicGen) GenerateWithImage(ctx context.Context, prompt string, image io.Reader, mimeType MimeType) (string, error) {
	return a.GenerateWithImages(ctx, prompt, []io.Reader{image}, []MimeType{mimeType})
}

func (a *AnthropicGen) GenerateWithImages(ctx context.Context, prompt string, images []io.Reader, mimeTypes []MimeType) (string, error) {
	if len(images) != len(mimeTypes) {
		return "", fmt.Errorf("number of images and mime types must match")
	}

	// Create a single chat message with the prompt and images
	msg := Message{
		Role:    RoleUser,
		Content: prompt,
	}

	// Add images to the message
	for i, image := range images {
		msg.Image = image
		msg.MimeType = mimeTypes[i]
	}

	// Use GenerateWithMessages with a single message
	return a.GenerateWithMessages(ctx, []Message{msg})
}

func (a *AnthropicGen) GenerateWithMessages(ctx context.Context, messages []Message) (string, error) {
	var anthropicMessages []anthropic.Message

	for _, msg := range messages {
		var contents []anthropic.MessageContent

		// Handle image if present
		if msg.Image != nil {
			imageBytes, err := io.ReadAll(msg.Image)
			if err != nil {
				return "", err
			}
			contents = append(contents, anthropic.NewImageMessageContent(
				anthropic.NewMessageContentSource(
					anthropic.MessagesContentSourceTypeBase64,
					string(msg.MimeType),
					imageBytes,
				),
			))
		}

		// Add text content
		if msg.Content != "" {
			contents = append(contents, anthropic.NewTextMessageContent(msg.Content))
		}

		anthropicMessages = append(anthropicMessages, anthropic.Message{
			Role:    anthropic.ChatRole(msg.Role),
			Content: contents,
		})
	}

	req := anthropic.MessagesRequest{
		Model:     anthropic.Model(a.model),
		Messages:  anthropicMessages,
		MaxTokens: a.maxTokens,
	}

	resp, err := a.client.CreateMessages(ctx, req)
	if err != nil {
		return "", err
	}

	return resp.Content[0].GetText(), nil
}
