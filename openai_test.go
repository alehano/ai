package ai

import (
	"context"
	"os"
	"testing"
)

func TestOpenAI(t *testing.T) {
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
		// OpenAIKey:         os.Getenv("OPENAI_API_KEY"),
		// OpenAIModel:       os.Getenv("OPENAI_MODEL"),
		// APIKey: os.Getenv("LAMBDA_LAB_API_KEY"),
		// Model:  os.Getenv("LAMBDA_LAB_MODEL"),
		// APIKey: os.Getenv("GROQ_API_KEY"),
		// Model:  os.Getenv("GROQ_MODEL"),
		APIKey:            os.Getenv("XAI_API_KEY"),
		Model:             os.Getenv("XAI_MODEL"),
		DefaultTokesLimit: 1000,
	}

	// llmGenOpenAI := NewOpenAIClient(cfg.OpenAIKey, cfg.OpenAIModel, int64(cfg.DefaultTokesLimit), 1.0, false)
	// llmGenOpenAI := NewGroqClient(cfg.APIKey, cfg.Model, int64(cfg.DefaultTokesLimit), 1.0, false)
	// llmGenOpenAI := NewLambdaLabClient(cfg.APIKey, cfg.Model, int64(cfg.DefaultTokesLimit), 1.0, false)
	llmGenOpenAI := NewXAI(cfg.APIKey, cfg.Model, int64(cfg.DefaultTokesLimit), 1.0, false)

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
