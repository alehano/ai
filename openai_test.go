package ai

import (
	"bytes"
	"context"
	"os"
	"testing"
)

func TestGenerateWithImage(t *testing.T) {
	imgData, err := os.ReadFile("test/test.webp")
	if err != nil {
		t.Fatalf("Error reading image: %v", err)
	}
	mimeType := "image/webp"

	cfg := struct {
		APIKey            string
		Model             string
		DefaultTokesLimit int
	}{
		APIKey: os.Getenv("GOOGLE_API_KEY"),
		Model:  os.Getenv("GOOGLE_MODEL"),
		// APIKey:            os.Getenv("OPENAI_API_KEY"),
		// Model:             os.Getenv("OPENAI_MODEL"),
		// APIKey:            os.Getenv("ANTHROPIC_API_KEY"),
		// Model:             os.Getenv("ANTHROPIC_MODEL"),
		DefaultTokesLimit: 4000,
	}

	llm := NewGoogleSimple(cfg.APIKey, cfg.Model, int64(cfg.DefaultTokesLimit), 1.0, false)
	// llm := NewOpenAI(cfg.APIKey, cfg.Model, int64(cfg.DefaultTokesLimit), 1.0, false)
	// llm := NewAnthropic(cfg.APIKey, cfg.Model, int(cfg.DefaultTokesLimit), 1.0, false)

	res, err := llm.GenerateWithImage(context.Background(), "describe the image", bytes.NewReader(imgData), MimeType(mimeType))
	if err != nil {
		t.Fatalf("Error generating from image: %v", err)
	}
	t.Logf("AI %s response: %v", llm.GetModel(), res)

}

func TestGenerateWithMessages(t *testing.T) {
	// Load test.jpg
	image, err := os.Open("test/test.jpg")
	if err != nil {
		t.Fatalf("Error loading image: %v", err)
	}
	defer image.Close()

	// Initialize cfg if not already done
	cfg := struct {
		APIKey            string
		Model             string
		DefaultTokesLimit int
	}{
		APIKey: os.Getenv("OPENAI_API_KEY"),
		Model:  os.Getenv("OPENAI_MODEL"),
		// APIKey: os.Getenv("LAMBDA_LAB_API_KEY"),
		// Model:  os.Getenv("LAMBDA_LAB_MODEL"),
		// APIKey: os.Getenv("GROQ_API_KEY"),
		// Model:  os.Getenv("GROQ_MODEL"),
		// APIKey:            os.Getenv("XAI_API_KEY"),
		// Model:             os.Getenv("XAI_MODEL"),

		// APIKey:            os.Getenv("GOOGLE_API_KEY"),
		// Model:             os.Getenv("GOOGLE_MODEL"),
		DefaultTokesLimit: 1000,
	}

	llmGenOpenAI := NewOpenAI(cfg.APIKey, cfg.Model, int64(cfg.DefaultTokesLimit), 1.0, false)
	// llmGenOpenAI := NewGroqClient(cfg.APIKey, cfg.Model, int64(cfg.DefaultTokesLimit), 1.0, false)
	// llmGenOpenAI := NewLambdaLabClient(cfg.APIKey, cfg.Model, int64(cfg.DefaultTokesLimit), 1.0, false)
	// llmGenOpenAI := NewGoogleSimple(cfg.APIKey, cfg.Model, int64(cfg.DefaultTokesLimit), 1.0, false)

	res, err := llmGenOpenAI.GenerateWithMessages(context.Background(), []Message{
		{
			Role:     RoleSystem,
			Image:    image,
			MimeType: MimeTypeJPEG,
		},
		{
			Role:    RoleUser,
			Content: "Is it healthy?",
		},
		{
			Role:    RoleAssistant,
			Content: "It may help protect against certain diseases.",
		},
		{
			Role:    RoleUser,
			Content: "Which ones for example?",
		},
	})
	if err != nil {
		t.Fatalf("Error generating from image: %v", err)
	}
	t.Logf("AI %s response: %v", llmGenOpenAI.GetModel(), res)

	// res, err := llmGenOpenAI.Generate(context.Background(), "You are a helpful assistant.", "What is the weather in Tokyo?")
	// if err != nil {
	// 	t.Fatalf("Error generating from image: %v", err)
	// }
	// t.Logf("AI %s response: %v", llmGenOpenAI.GetModel(), res)
}
