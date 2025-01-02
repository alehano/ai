package ai

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"strings"

	"cloud.google.com/go/vertexai/genai"
	"google.golang.org/api/iterator"
	"google.golang.org/api/option"
)

type Google struct {
	client         *genai.Client
	model          string
	safetySettings []*genai.SafetySetting
	maxTokens      int
	temperature    *float32
}

const maxImageSize = 4 * 1024 * 1024 // 4MB

func validateImageSize(image io.Reader) (io.Reader, error) {
	// Read the image into a buffer while checking size
	var buf bytes.Buffer
	lr := io.LimitReader(image, maxImageSize+1)
	n, err := io.Copy(&buf, lr)
	if err != nil {
		return nil, err
	}
	if n > maxImageSize {
		return nil, fmt.Errorf("image exceeds maximum size of %d bytes", maxImageSize)
	}
	// Return a new reader with the buffered content
	return bytes.NewReader(buf.Bytes()), nil
}

func NewGoogle(projectID, location, model string, maxTokens int, temperature *float32, opts ...option.ClientOption) (*Google, error) {
	client, err := genai.NewClient(context.Background(), projectID, location, opts...)
	if err != nil {
		return nil, fmt.Errorf("failed to create Google client: %v", err)
	}
	return &Google{
		client:      client,
		model:       model,
		maxTokens:   maxTokens,
		temperature: temperature,
	}, nil
}

func (g *Google) Generate(ctx context.Context, systemPrompt, prompt string) (string, error) {
	gModel := g.client.GenerativeModel(g.model)
	gModel.SafetySettings = g.safetySettings
	if g.temperature != nil {
		gModel.Temperature = g.temperature
	}
	gModel.GenerationConfig.SetMaxOutputTokens(int32(g.maxTokens))
	gModel.SystemInstruction = &genai.Content{
		Parts: []genai.Part{genai.Text(systemPrompt)},
	}

	resp, err := gModel.GenerateContent(ctx, genai.Text(prompt))
	if err != nil {
		return "", fmt.Errorf("failed to generate content: %v", err)
	}

	if len(resp.Candidates) == 0 || resp.Candidates[0].Content == nil {
		return "", fmt.Errorf("no content generated")
	}

	var res strings.Builder
	if len(resp.Candidates) > 0 && resp.Candidates[0] != nil && resp.Candidates[0].Content != nil &&
		len(resp.Candidates[0].Content.Parts) > 0 {
		for _, part := range resp.Candidates[0].Content.Parts {
			res.WriteString(fmt.Sprintf("%v", part))
		}
	}
	return res.String(), nil
}

func (g *Google) GenerateStream(ctx context.Context, systemPrompt, prompt string, resultCh chan string, doneCh chan bool, errCh chan error) {
	gModel := g.client.GenerativeModel(g.model)
	gModel.SafetySettings = g.safetySettings
	if g.temperature != nil {
		gModel.Temperature = g.temperature
	}
	gModel.GenerationConfig.SetMaxOutputTokens(int32(g.maxTokens))
	gModel.SystemInstruction = &genai.Content{
		Parts: []genai.Part{genai.Text(systemPrompt)},
	}

	iter := gModel.GenerateContentStream(ctx, genai.Text(prompt))

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

func (g *Google) GetModel() string {
	return g.model
}

func (g *Google) GenerateWithImage(ctx context.Context, prompt string, image io.Reader, mimeType MimeType) (string, error) {
	return g.GenerateWithImages(ctx, prompt, []io.Reader{image}, []MimeType{mimeType})
}

func (g *Google) GenerateWithImages(ctx context.Context, prompt string, images []io.Reader, mimeTypes []MimeType) (string, error) {
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

func (g *Google) GenerateWithMessages(ctx context.Context, messages []Message) (string, error) {
	gModel := g.client.GenerativeModel(g.model)
	gModel.SafetySettings = g.safetySettings
	if g.temperature != nil {
		gModel.Temperature = g.temperature
	}
	gModel.GenerationConfig.SetMaxOutputTokens(int32(g.maxTokens))
	// Start chat and set history
	cs := gModel.StartChat()

	// Convert ChatMessages to genai.Content with roles
	var history []*genai.Content
	for _, msg := range messages {

		if msg.Role == RoleSystem {
			gModel.SystemInstruction = &genai.Content{
				Parts: []genai.Part{genai.Text(msg.Content)},
			}
			continue
		}

		var parts []genai.Part

		if msg.Image != nil {
			// Validate and read image data
			validatedImage, err := validateImageSize(msg.Image)
			if err != nil {
				return "", err
			}
			imageData, err := io.ReadAll(validatedImage)
			if err != nil {
				return "", fmt.Errorf("failed to read image: %v", err)
			}

			// Get the correct format from MIME type
			format := strings.TrimPrefix(string(msg.MimeType), "image/")
			parts = append(parts, genai.ImageData(format, imageData))
		}

		// Add text content
		if msg.Content != "" {
			parts = append(parts, genai.Text(msg.Content))
		}

		// Create content with role
		history = append(history, &genai.Content{
			Parts: parts,
			Role:  convertRole(msg.Role),
		})
	}

	// Set chat history
	cs.History = history

	// Send message (use the last message as the prompt)
	if len(messages) == 0 {
		return "", fmt.Errorf("no messages provided")
	}
	lastMessage := messages[len(messages)-1]

	// Generate response
	resp, err := cs.SendMessage(ctx, genai.Text(lastMessage.Content))
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

func convertRole(role Role) string {
	switch role {
	case RoleSystem:
		return "user"
	case RoleUser:
		return "user"
	case RoleAssistant:
		return "model"
	}
	return "user"
}
