package ai

import (
	"context"
	"io"
)

type MimeType string

const (
	MimeTypePNG  MimeType = "image/png"
	MimeTypeJPEG MimeType = "image/jpeg"
	MimeTypeWEBP MimeType = "image/webp"
	MimeTypeHEIC MimeType = "image/heic"
	MimeTypeHEIF MimeType = "image/heif"
)

type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
)

// LLMChat defines the interface for chat (multi-message)
type Message struct {
	Role     Role
	Image    io.Reader // optional
	MimeType MimeType  // optional
	Content  string    // optional
}

// LLMGen defines the interface for language model generators
type LLMGen interface {
	// Generate produces a response given a system prompt and user prompt
	Generate(ctx context.Context, systemPrompt, prompt string) (string, error)

	// GenerateStream streams the generated response
	GenerateStream(ctx context.Context, systemPrompt, prompt string, resultCh chan string, doneCh chan bool, errCh chan error)

	// GetModel returns the name of the current model
	GetModel() string

	// GenerateWithImage generates text from an image
	GenerateWithImage(ctx context.Context, prompt string, image io.Reader, mimeType MimeType) (string, error)

	// GenerateWithImages generates text from multiple images
	GenerateWithImages(ctx context.Context, prompt string, images []io.Reader, mimeTypes []MimeType) (string, error)

	GenerateWithMessages(ctx context.Context, messages []Message) (string, error)
}
