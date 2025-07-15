import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Sample dataset with spam and non-spam messages
data = {
    'text': [
        'Congratulations! You have won a lottery of $1000',
        'Click here to claim your prize',
        'Hello, how are you doing today?',
        'Important update regarding your account',
        'You have a new message from your friend',
        'Get paid for taking surveys online',
        'This is a reminder for your appointment tomorrow',
        'You won a free ticket to the concert!',
        'Your subscription has been renewed',
        'Earn money from home with this simple trick',
        'Hi, just checking in to see how you are doing.',
        'Don’t miss out on this exclusive offer!',
        'Let’s catch up soon!',
        'This is not a spam message.',
        'You have a new follower on social media.',
        'Limited time offer! Act now!',
        'Your account has been compromised. Click here to secure it.',
        'I hope you are having a great day!',
        'This is a legitimate message, no spam here.',
        'You have been selected for a special promotion!'
    ],
    'label': [
        'spam', 'spam', 'ham', 'ham', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam',
        'ham', 'spam', 'ham', 'ham', 'ham', 'spam', 'spam', 'ham', 'ham', 'spam'
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Convert text data into numerical vectors
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Make predictions
y_pred = model.predict(X_test_vectorized)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['ham', 'spam'])

print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(report)
