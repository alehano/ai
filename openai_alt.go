package ai

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"io"

	openai "github.com/sashabaranov/go-openai"
)

// Alternative OpenAI client with unoffical lib
// Deprecated: use OpenAIClient instead
type OpenAIAltGen struct {
	client      *openai.Client
	model       string
	maxTokens   int
	temperature float32
	isJson      bool
}

func NewOpenAIAltGen(apiKey, model string, maxTokens int, temperature float32, isJson bool) *OpenAIAltGen {
	client := openai.NewClient(apiKey)

	return &OpenAIAltGen{
		client:      client,
		model:       model,
		maxTokens:   maxTokens,
		temperature: temperature,
		isJson:      isJson,
	}
}

func (o *OpenAIAltGen) Generate(ctx context.Context, systemPrompt, prompt string) (string, error) {
	messages := []openai.ChatCompletionMessage{
		{
			Role:    openai.ChatMessageRoleUser,
			Content: prompt,
		},
	}

	if systemPrompt != "" {
		messages = append(messages, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleSystem,
			Content: systemPrompt,
		})
	}

	req := openai.ChatCompletionRequest{
		Model:       o.model,
		Messages:    messages,
		MaxTokens:   o.maxTokens,
		Temperature: o.temperature,
	}

	if o.isJson {
		req.ResponseFormat = &openai.ChatCompletionResponseFormat{
			Type: openai.ChatCompletionResponseFormatTypeJSONObject,
		}
	} else {
		req.ResponseFormat = &openai.ChatCompletionResponseFormat{
			Type: openai.ChatCompletionResponseFormatTypeText,
		}
	}

	resp, err := o.client.CreateChatCompletion(
		ctx,
		req,
	)

	if err != nil {
		return "", err
	}

	if len(resp.Choices) == 0 {
		return "", errors.New("no choices returned")
	}

	return resp.Choices[0].Message.Content, nil
}

func (o *OpenAIAltGen) GenerateStream(ctx context.Context, systemPrompt, prompt string, resultCh chan string, doneCh chan bool, errCh chan error) {
	messages := []openai.ChatCompletionMessage{
		{
			Role:    openai.ChatMessageRoleUser,
			Content: prompt,
		},
	}

	if systemPrompt != "" {
		messages = append(messages, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleSystem,
			Content: systemPrompt,
		})
	}

	stream, err := o.client.CreateChatCompletionStream(
		ctx,
		openai.ChatCompletionRequest{
			Model:       o.model,
			Messages:    messages,
			MaxTokens:   o.maxTokens,
			Temperature: o.temperature,
			Stream:      true,
		},
	)

	if err != nil {
		select {
		case errCh <- err:
		case <-ctx.Done():
		}
		return
	}
	defer stream.Close()

	for {
		select {
		case <-ctx.Done():
			// Context cancelled, stop generation
			return
		default:
			response, err := stream.Recv()
			if errors.Is(err, io.EOF) {
				select {
				case doneCh <- true:
				case <-ctx.Done():
				}
				return
			}
			if err != nil {
				select {
				case errCh <- err:
				case <-ctx.Done():
				}
				return
			}

			select {
			case resultCh <- response.Choices[0].Delta.Content:
			case <-ctx.Done():
				return
			}
		}
	}
}

func (o *OpenAIAltGen) GetModel() string {
	return o.model
}

func (o *OpenAIAltGen) GenerateWithImage(ctx context.Context, prompt string, image io.Reader, mimeType MimeType) (string, error) {
	return o.GenerateWithImages(ctx, prompt, []io.Reader{image}, []MimeType{mimeType})
}

func (o *OpenAIAltGen) GenerateWithImages(ctx context.Context, prompt string, images []io.Reader, mimeTypes []MimeType) (string, error) {
	if len(images) != len(mimeTypes) {
		return "", fmt.Errorf("number of images and mime types must match")
	}

	msg := Message{
		Role:    openai.ChatMessageRoleUser,
		Content: prompt,
	}

	for i, image := range images {
		msg.Image = image
		msg.MimeType = mimeTypes[i]
	}

	return o.GenerateWithMessages(ctx, []Message{msg})
}

func (o *OpenAIAltGen) GenerateWithMessages(ctx context.Context, messages []Message) (string, error) {
	var chatMessages []openai.ChatCompletionMessage

	for _, msg := range messages {
		message := openai.ChatCompletionMessage{
			Role: string(msg.Role),
		}

		if msg.Image != nil {
			imageBytes, err := io.ReadAll(msg.Image)
			if err != nil {
				return "", err
			}
			base64Image := base64.StdEncoding.EncodeToString(imageBytes)

			message.MultiContent = []openai.ChatMessagePart{
				{
					Type: openai.ChatMessagePartTypeText,
					Text: msg.Content,
				},
				{
					Type: openai.ChatMessagePartTypeImageURL,
					ImageURL: &openai.ChatMessageImageURL{
						URL: "data:" + string(msg.MimeType) + ";base64," + base64Image,
					},
				},
			}
		} else {
			message.Content = msg.Content
		}

		chatMessages = append(chatMessages, message)
	}

	req := openai.ChatCompletionRequest{
		Model:       o.model,
		Messages:    chatMessages,
		MaxTokens:   o.maxTokens,
		Temperature: o.temperature,
	}

	if o.isJson {
		req.ResponseFormat = &openai.ChatCompletionResponseFormat{
			Type: openai.ChatCompletionResponseFormatTypeJSONObject,
		}
	}

	resp, err := o.client.CreateChatCompletion(ctx, req)
	if err != nil {
		return "", err
	}

	if len(resp.Choices) == 0 {
		return "", errors.New("no choices returned")
	}

	return resp.Choices[0].Message.Content, nil
}
