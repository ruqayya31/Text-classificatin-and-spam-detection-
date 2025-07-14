import re
from sklearn.metrics import classification_report, accuracy_score

# Sample sentences for classification
sentences = [
    "I love programming in Python.",
    "The weather is nice today.",
    "I hate getting up early in the morning.",
    "This movie was fantastic!",
    "I dislike traffic jams.",
    "I enjoy hiking and outdoor activities.",
    "The food was okay, nothing special.",
    "I am feeling great today!",
    "I am not happy with the service.",
    "It was a boring experience."
]

# True labels for the sentences
true_labels = [
    "positive", "neutral", "negative", "positive", "negative",
    "positive", "neutral", "positive", "negative", "negative"
]

# Define keywords for classification
positive_keywords = ["love", "great", "fantastic", "enjoy", "happy", "good", "excellent"]
negative_keywords = ["hate", "dislike", "not happy", "boring", "bad", "terrible", "awful"]

# Function to classify sentences
def classify_sentence(sentence):
    sentence_lower = sentence.lower()

    # Count positive and negative keywords
    positive_count = sum(1 for word in positive_keywords if re.search(r'\b' + re.escape(word) + r'\b', sentence_lower))
    negative_count = sum(1 for word in negative_keywords if re.search(r'\b' + re.escape(word) + r'\b', sentence_lower))

    # Determine classification
    if positive_count > negative_count:
        return "positive"
    elif negative_count > positive_count:
        return "negative"
    else:
        return "neutral"

# Classify each sentence and store predictions
predicted_labels = [classify_sentence(sentence) for sentence in sentences]

# Calculate accuracy
accuracy = accuracy_score(true_labels, predicted_labels)

# Print classification report
print("Classification Report:")
print(classification_report(true_labels, predicted_labels))

# Print accuracy
print(f'Accuracy: {accuracy:.2f}\n')

# Print sample predictions
print("\nSample Predictions:")
for sentence, true, pred in zip(sentences, true_labels, predicted_labels):
    print(f'Text: "{sentence}"\nTrue: {true} | Predicted: {pred}\n')
