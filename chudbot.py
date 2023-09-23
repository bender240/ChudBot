from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
import nltk

# Ensure NLTK data is downloaded
nltk.download('punkt')

# Function to read training data from the "advfinal.txt" file
def read_training_data(file_path):
    training_data = []
    with open(file_path, 'r') as file:
        for line in file:
            # Each line should contain a conversation pair separated by a comma
            parts = line.strip().split(',')
            if len(parts) == 2:
                training_data.append(f'{parts[0].strip()}, {parts[1].strip()}')
    return training_data

# Path to your training data text file
training_file_path = 'advfinal.txt'  # Replace with your file path

# Read the training data from the "advfinal.txt" file
train_data = read_training_data(training_file_path)

# Initialize the chatbot
chatbot = ChatBot("chud")

# Create a new ListTrainer and train the chatbot
trainer = ListTrainer(chatbot)

# Train the chatbot
trainer.train(train_data)

exit_conditions = (":q", "quit", "exit")

while True:
    query = input("> ")

    if query in exit_conditions:
        break
    else:
        response = chatbot.get_response(query)
        print(f"🪴 {response}")
