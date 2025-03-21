package ai

import (
	"bytes"
	"context"
	"os"
	"testing"
)

func TestGoogleGenerateWithImage(t *testing.T) {
	imgData, err := os.ReadFile("test/test.webp")
	if err != nil {
		t.Fatalf("Error reading image: %v", err)
	}
	mimeType := "image/webp"

	temp := float32(1.0)
	llm, err := NewGoogle(os.Getenv("GOOGLE_PROJECT_ID"), []string{os.Getenv("GOOGLE_LOCATION")},
		os.Getenv("GOOGLE_MODEL"), 4000, &temp, false)
	if err != nil {
		t.Fatalf("Error creating Google client: %v", err)
	}

	// res, err := llm.GenerateWithImage(context.Background(), "describe the image", bytes.NewReader(imgData), MimeType(mimeType))
	res, err := llm.GenerateWithImage(context.Background(), "make it green", bytes.NewReader(imgData), MimeType(mimeType))
	if err != nil {
		t.Fatalf("Error generating from image: %v", err)
	}
	t.Logf("AI %s response: %v", llm.GetModel(), res)

}
