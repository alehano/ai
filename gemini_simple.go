package ai

import (
	"context"
	"errors"
	"fmt"
	"io"
	"strings"

	"github.com/google/generative-ai-go/genai"
	"google.golang.org/api/iterator"
	"google.golang.org/api/option"
)

// GeminiSimpleLLM is a simple Gemini client that uses the official Google Gemini API
// Deprecated: use GeminiClient (open ai compatible) instead
type GeminiSimpleLLM struct {
	apiKey      string
	model       string
	maxTokens   int
	isJSON      bool
	temperature *float32
}

func NewGeminiSimpleOld(apiKey, model string, maxTokens int, isJSON bool, temperature *float32) *GeminiSimpleLLM {
	return &GeminiSimpleLLM{
		apiKey:      apiKey,
		model:       model,
		maxTokens:   maxTokens,
		isJSON:      isJSON, // https://ai.google.dev/gemini-api/docs/structured-output?lang=go
		temperature: temperature,
	}
}

func (g *GeminiSimpleLLM) Generate(ctx context.Context, systemPrompt, prompt string) (string, error) {
	client, err := genai.NewClient(ctx, option.WithAPIKey(g.apiKey))
	if err != nil {
		return "", fmt.Errorf("failed to create Gemini client: %v", err)
	}
	defer client.Close()

	model := client.GenerativeModel(g.model)
	if g.temperature != nil {
		model.Temperature = g.temperature
	}
	if g.isJSON {
		model.ResponseMIMEType = "application/json"
	}
	model.GenerationConfig.SetMaxOutputTokens(int32(g.maxTokens))
	model.SystemInstruction = &genai.Content{
		Parts: []genai.Part{genai.Text(systemPrompt)},
	}

	resp, err := model.GenerateContent(ctx, genai.Text(prompt))
	if err != nil {
		return "", fmt.Errorf("failed to generate content: %v", err)
	}

	if len(resp.Candidates) == 0 || resp.Candidates[0].Content == nil {
		return "", fmt.Errorf("no content generated")
	}

	fmt.Printf("resp: %+v", resp.Candidates[0].TokenCount)

	var res strings.Builder
	if len(resp.Candidates) > 0 && resp.Candidates[0].Content != nil &&
		len(resp.Candidates[0].Content.Parts) > 0 {
		for _, part := range resp.Candidates[0].Content.Parts {
			res.WriteString(fmt.Sprintf("%v", part))
		}
	}
	return res.String(), nil
}

// TODO: test it
func (g *GeminiSimpleLLM) GenerateStream(ctx context.Context, systemPrompt, prompt string, resultCh chan string, doneCh chan bool, errCh chan error) {
	client, err := genai.NewClient(ctx, option.WithAPIKey(g.apiKey))
	if err != nil {
		errCh <- fmt.Errorf("failed to create Gemini client: %v", err)
		return
	}
	defer client.Close()

	model := client.GenerativeModel(g.model)
	if g.temperature != nil {
		model.Temperature = g.temperature
	}
	model.GenerationConfig.SetMaxOutputTokens(int32(g.maxTokens))
	model.SystemInstruction = &genai.Content{
		Parts: []genai.Part{genai.Text(systemPrompt)},
	}

	iter := model.GenerateContentStream(ctx, genai.Text(prompt))

	go func() {
		for {
			select {
			case <-ctx.Done():
				errCh <- ctx.Err()
				return
			default:
				resp, err := iter.Next()
				if err != nil {
					if errors.Is(err, iterator.Done) {
						select {
						case doneCh <- true:
						case <-ctx.Done():
						}
						return
					}
					select {
					case errCh <- fmt.Errorf("error in stream: %v", err):
					case <-ctx.Done():
					}
					return
				}

				if len(resp.Candidates) > 0 && resp.Candidates[0].Content != nil {
					for _, part := range resp.Candidates[0].Content.Parts {
						if text, ok := part.(genai.Text); ok {
							select {
							case resultCh <- string(text):
							case <-ctx.Done():
								return
							}
						}
					}
				}
			}
		}
	}()
}

func (g *GeminiSimpleLLM) GetModel() string {
	return g.model
}

func (g *GeminiSimpleLLM) GenerateWithImage(ctx context.Context, prompt string, image io.Reader, mimeType MimeType) (string, error) {
	return g.GenerateWithImages(ctx, prompt, []io.Reader{image}, []MimeType{mimeType})
}

func (g *GeminiSimpleLLM) GenerateWithImages(ctx context.Context, prompt string, images []io.Reader, mimeTypes []MimeType) (string, error) {
	if len(images) != len(mimeTypes) {
		return "", fmt.Errorf("number of images and mime types must match")
	}

	// Create a single chat message with the prompt and images
	msg := Message{
		Content: prompt,
	}

	// Add images to the message
	for i, image := range images {
		msg.Image = image
		msg.MimeType = mimeTypes[i]
	}

	// Use GenerateWithMessages with a single message
	return g.GenerateWithMessages(ctx, []Message{msg})
}

func (g *GeminiSimpleLLM) GenerateWithMessages(ctx context.Context, messages []Message) (string, error) {
	client, err := genai.NewClient(ctx, option.WithAPIKey(g.apiKey))
	if err != nil {
		return "", fmt.Errorf("failed to create Gemini client: %v", err)
	}
	defer client.Close()

	model := client.GenerativeModel(g.model)
	if g.temperature != nil {
		model.Temperature = g.temperature
	}
	if g.isJSON {
		model.ResponseMIMEType = "application/json"
	}
	model.GenerationConfig.SetMaxOutputTokens(int32(g.maxTokens))

	// Prepare chat history and current message parts
	var parts []genai.Part

	for _, msg := range messages {
		if msg.Image != nil {
			imageData, err := io.ReadAll(msg.Image)
			if err != nil {
				return "", fmt.Errorf("failed to read image: %v", err)
			}
			parts = append(parts, genai.ImageData(string(msg.MimeType), imageData))
		}

		// Add text content
		if msg.Content != "" {
			parts = append(parts, genai.Text(msg.Content))
		}
	}

	// Generate response
	resp, err := model.GenerateContent(ctx, parts...)
	if err != nil {
		return "", fmt.Errorf("failed to generate chat content: %v", err)
	}

	if len(resp.Candidates) == 0 || resp.Candidates[0].Content == nil {
		return "", fmt.Errorf("no content generated")
	}

	var res strings.Builder
	for _, part := range resp.Candidates[0].Content.Parts {
		res.WriteString(fmt.Sprintf("%v", part))
	}
	return res.String(), nil
}
