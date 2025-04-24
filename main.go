package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log"
	"mime/multipart"
	"net/http"
	"strings"
	"time"

	"cloud.google.com/go/firestore"
	"github.com/google/uuid"
	"google.golang.org/api/option"
)

type DetectionResponse struct {
	UserId        string `json:"user_id"`
	Query         string `json:"query"`
	GenAiResponse string `json:"gen_ai_response"`
}

type OllamaResponse struct {
	Response string `json:"response"`
	Done     bool   `json:"done"`
}

var firestoreClient *firestore.Client

func main() {
	ctx := context.Background()

	sa := option.WithCredentialsFile("leafscan-d0ee4-firebase-adminsdk-fbsvc-cb14153170.json")
	client, err := firestore.NewClient(ctx, "leafscan-d0ee4", sa)
	if err != nil {
		log.Fatalf("Failed to create Firestore client: %v", err)
	}
	defer client.Close()
	firestoreClient = client

	http.HandleFunc("/detect", detectHandler)
	log.Println("Server running at :8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}

func detectHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

	if r.Method != http.MethodPost {
		http.Error(w, "Only POST allowed", http.StatusMethodNotAllowed)
		return
	}

	err := r.ParseMultipartForm(10 << 20)
	if err != nil {
		http.Error(w, "Invalid form: "+err.Error(), http.StatusBadRequest)
		return
	}

	userId := r.FormValue("user_id")
	if userId == "" {
		http.Error(w, "Missing user_id", http.StatusBadRequest)
		return
	}

	file, _, err := r.FormFile("image")
	if err != nil {
		http.Error(w, "Image missing: "+err.Error(), http.StatusBadRequest)
		return
	}
	defer file.Close()

	imageData, err := readFileData(file)
	if err != nil {
		http.Error(w, "Image read failed: "+err.Error(), http.StatusInternalServerError)
		return
	}

	leafName, err := requestFlaskDetectionServer(imageData)
	if err != nil {
		http.Error(w, "Flask error: "+err.Error(), http.StatusInternalServerError)
		return
	}

	prompt := leafName + " Give 4 small points about medicinal use cases no bold words and max 8 words and no additional things."
	aiResponse, err := ollamaInformationModel(prompt)
	if err != nil {
		http.Error(w, "Ollama error: "+err.Error(), http.StatusInternalServerError)
		return
	}

	chatID := uuid.New().String()
	err = storeInFirestore(userId, chatID, leafName, aiResponse)
	if err != nil {
		http.Error(w, "Firestore error: "+err.Error(), http.StatusInternalServerError)
		return
	}

	resp := DetectionResponse{
		UserId:        userId,
		Query:         leafName,
		GenAiResponse: aiResponse,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func readFileData(file multipart.File) ([]byte, error) {
	const maxFileSize = 10 * 1024 * 1024
	limitedReader := io.LimitReader(file, maxFileSize)
	return io.ReadAll(limitedReader)
}

func requestFlaskDetectionServer(imageData []byte) (string, error) {
	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)

	formFile, err := writer.CreateFormFile("file", "upload.jpg")
	if err != nil {
		return "", err
	}
	if _, err := formFile.Write(imageData); err != nil {
		return "", err
	}
	writer.Close()

	req, err := http.NewRequest("POST", "http://localhost:5000/predict", &buf)
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", writer.FormDataContentType())

	client := &http.Client{}
	res, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer res.Body.Close()

	if res.StatusCode != http.StatusOK {
		return "", errors.New("Flask returned " + res.Status)
	}

	body, err := io.ReadAll(res.Body)
	if err != nil {
		return "", err
	}

	var result map[string]interface{}
	if err := json.Unmarshal(body, &result); err != nil {
		return "", err
	}

	label, ok := result["predicted_label"].(string)
	if !ok {
		return "", errors.New("No label returned")
	}

	return label, nil
}

func ollamaInformationModel(prompt string) (string, error) {
	body, err := json.Marshal(map[string]string{
		"model":  "llama3.2",
		"prompt": prompt,
	})
	if err != nil {
		return "", err
	}

	res, err := http.Post("http://localhost:11434/api/generate", "application/json", bytes.NewBuffer(body))
	if err != nil {
		return "", err
	}
	defer res.Body.Close()

	var fullResponse strings.Builder
	decoder := json.NewDecoder(res.Body)

	for {
		var chunk OllamaResponse
		if err := decoder.Decode(&chunk); err == io.EOF {
			break
		} else if err != nil {
			return "", err
		}

		fullResponse.WriteString(chunk.Response)

		if chunk.Done {
			break
		}
	}

	return fullResponse.String(), nil
}

func storeInFirestore(userId, chatID, plantName, message string) error {
	ctx := context.Background()

	chatRef := firestoreClient.
		Collection("chats").
		Doc(userId).
		Collection("chat").
		Doc(chatID)

	// Create chat metadata with plant name
	_, err := chatRef.Set(ctx, map[string]interface{}{
		"name": plantName,
	})
	if err != nil {
		return err
	}

	// Add the message to "messages" subcollection
	_, err = chatRef.Collection("messages").NewDoc().Set(ctx, map[string]interface{}{
		"message":   message,
		"timestamp": time.Now(),
		"user_id":   userId,
		"user_type": "AI",
	})
	return err
}
