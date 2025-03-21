package ai

import (
	"context"
	"encoding/base64"
	"fmt"
	"io"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

type OpenAI struct {
	client      *openai.Client
	model       string
	maxTokens   int64
	temperature float64
	isJson      bool
}

func NewOpenAI(apiKey string, model string, maxTokens int64, temperature float64, isJson bool) *OpenAI {
	return NewOpenAICompatible("https://api.openai.com/v1/", apiKey, model, maxTokens, temperature, isJson)
}

func NewGoogleSimple(apiKey string, model string, maxTokens int64, temperature float64, isJson bool) *OpenAI {
	return NewOpenAICompatible("https://generativelanguage.googleapis.com/v1beta/openai/", apiKey, model, maxTokens, temperature, isJson)
}

// https://docs.lambdalabs.com/public-cloud/lambda-inference-api/
// Caution: Do not works with images
func NewLambdaLab(apiKey string, model string, maxTokens int64, temperature float64, isJson bool) *OpenAI {
	return NewOpenAICompatible("https://api.lambdalabs.com/v1/", apiKey, model, maxTokens, temperature, isJson)
}

// https://console.groq.com/docs/
// Not working, not fully compatible with OpenAI API
// func NewGroqClient(apiKey string, model string, maxTokens int64, temperature float64, isJson bool) *OpenAIClient {
// return NewOpenAICompatibleClient("https://api.groq.com/openai/v1/", apiKey, model, maxTokens, temperature, isJson)
// }

// https://docs.x.ai/docs/api-reference
func NewXAI(apiKey string, model string, maxTokens int64, temperature float64, isJson bool) *OpenAI {
	return NewOpenAICompatible("https://api.x.ai/v1/", apiKey, model, maxTokens, temperature, isJson)
}

func NewOpenAICompatible(baseURL, apiKey string, model string, maxTokens int64, temperature float64, isJson bool) *OpenAI {
	client := openai.NewClient(
		option.WithAPIKey(apiKey),
		option.WithBaseURL(baseURL),
	)
	return &OpenAI{
		client:      client,
		model:       model,
		maxTokens:   maxTokens,
		temperature: temperature,
		isJson:      isJson,
	}
}

func (o *OpenAI) Generate(ctx context.Context, systemPrompt, prompt string) (string, error) {
	params := openai.ChatCompletionNewParams{
		Messages: openai.F([]openai.ChatCompletionMessageParamUnion{
			openai.SystemMessage(systemPrompt),
			openai.UserMessage(prompt),
		}),
		Model:       openai.F(o.model),
		MaxTokens:   openai.F(o.maxTokens),
		Temperature: openai.F(o.temperature),
	}

	if o.isJson {
		params.ResponseFormat = openai.F[openai.ChatCompletionNewParamsResponseFormatUnion](
			openai.ResponseFormatJSONObjectParam{
				Type: openai.F(openai.ResponseFormatJSONObjectTypeJSONObject),
			},
		)
	}

	completion, err := o.client.Chat.Completions.New(ctx, params)
	if err != nil {
		return "", err
	}
	return completion.Choices[0].Message.Content, nil
}

func (o *OpenAI) GenerateStream(ctx context.Context, systemPrompt, prompt string, resultCh chan string, doneCh chan bool, errCh chan error) {
	stream := o.client.Chat.Completions.NewStreaming(ctx, openai.ChatCompletionNewParams{
		Messages: openai.F([]openai.ChatCompletionMessageParamUnion{
			openai.SystemMessage(systemPrompt),
			openai.UserMessage(prompt),
		}),
		Model: openai.F(o.model),
	})

	go func() {
		defer close(resultCh)
		defer close(doneCh)
		defer close(errCh)

		for stream.Next() {
			chunk := stream.Current()
			if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != "" {
				resultCh <- chunk.Choices[0].Delta.Content
			}
		}

		if err := stream.Err(); err != nil {
			errCh <- err
			return
		}
		doneCh <- true
	}()
}

func (o *OpenAI) GetModel() string {
	return o.model
}

func (o *OpenAI) GenerateWithImage(ctx context.Context, prompt string, image io.Reader, mimeType MimeType) (string, error) {
	return o.GenerateWithImages(ctx, prompt, []io.Reader{image}, []MimeType{mimeType})
}

func (o *OpenAI) GenerateWithImages(ctx context.Context, prompt string, images []io.Reader, mimeTypes []MimeType) (string, error) {
	if len(images) != len(mimeTypes) {
		return "", fmt.Errorf("number of images and mime types must match")
	}

	if prompt == "" {
		return "", fmt.Errorf("prompt is required")
	}

	msgs := []Message{}

	// Add images to the message
	for i, image := range images {
		msgs = append(msgs, Message{
			Role:     RoleUser,
			Image:    image,
			MimeType: mimeTypes[i],
		})
	}

	msgs = append(msgs, Message{
		Role:    RoleUser,
		Content: prompt,
	})

	// Use GenerateWithMessages with a single message
	return o.GenerateWithMessages(ctx, msgs)
}

func (o *OpenAI) GenerateWithMessages(ctx context.Context, messages []Message) (string, error) {
	chatMessages := make([]openai.ChatCompletionMessageParamUnion, len(messages))

	for i, msg := range messages {
		if msg.Image != nil {
			// Convert image to base64
			imageData, err := io.ReadAll(msg.Image)
			if err != nil {
				return "", err
			}
			base64Image := base64.StdEncoding.EncodeToString(imageData)

			// Create message with both text and image
			chatMessages[i] = openai.UserMessageParts(
				openai.ImagePart("data:" + string(msg.MimeType) + ";base64," + base64Image),
			)
		} else {
			// Regular text message
			switch msg.Role {
			case RoleUser:
				chatMessages[i] = openai.UserMessage(msg.Content)
			case RoleAssistant:
				chatMessages[i] = openai.AssistantMessage(msg.Content)
			case RoleSystem:
				chatMessages[i] = openai.SystemMessage(msg.Content)
			}
		}
	}

	params := openai.ChatCompletionNewParams{
		Model:       openai.F(o.model),
		Messages:    openai.F(chatMessages),
		MaxTokens:   openai.F(o.maxTokens),
		Temperature: openai.F(o.temperature),
	}

	if o.isJson {
		params.ResponseFormat = openai.F[openai.ChatCompletionNewParamsResponseFormatUnion](
			openai.ResponseFormatJSONObjectParam{
				Type: openai.F(openai.ResponseFormatJSONObjectTypeJSONObject),
			},
		)
	}

	resp, err := o.client.Chat.Completions.New(ctx, params)
	if err != nil {
		return "", err
	}
	return resp.Choices[0].Message.Content, nil
}
