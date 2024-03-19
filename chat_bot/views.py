import numpy as np
from django.shortcuts import render
from django.http import HttpResponse, HttpResponseBadRequest, JsonResponse
import nltk
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
import json
import numpy as np
import torch
import random
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from .intents import intents
# Initialize lemmatizer
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation and no softmax at the end
        return out
def tokenize(sentence):
    """
    Split sentence into array of words/tokens
    """
    return nltk.word_tokenize(sentence)

def stem(word):
    """
    stemming = find the root form of the word
    examples:
    words = ["organize", "organizes", "organizing"]
    words = [stem(w) for w in words]
    -> ["organ", "organ", "organ"]
    """
    return stemmer.stem(word.lower())
def bag_of_words(tokenized_sentence, words):
    """
    Return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    """
    # Lemmatize each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # Initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1
    return bag
#with open('intents', 'r') as f:
    intents = json.load(f)
all_words = []
tags = []
xy = []
for intent in intents:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
      w = tokenize(pattern)
      all_words.extend(w)
      xy.append((w, tag))
ignore_words = ['?', '.', '!']
all_words=[stem(w)for w in all_words if  w not in ignore_words]
all_words=sorted(set(all_words))
tags=sorted(set(tags))
x_train=[]
y_train=[]
for (pattern_sentece,tag)in xy:
    bag=bag_of_words(pattern_sentece,all_words)
    x_train.append(bag)
    label=tags.index(tag)
    y_train.append(label)
x_train=np.array(x_train)
y_train=np.array(y_train)
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples =len(x_train)
        self.x_data = x_train
        self.y_data=y_train
    #dataset[idx]
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
        # we can call len(dataset) to return the size

    def __len__(self):
        return self.n_samples
batch_size=8
hidden_size=8
input_size=len(x_train[5])
output_size=len(tags)
learning_rate=0.001
num_epochs=1500
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)
device=torch.device('cuda'if torch.cuda.is_available()else 'cpu')
model = NeuralNet(input_size,
                  hidden_size,
                  output_size)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device, dtype=torch.long)
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch+1) % 100 == 0:
        print (f'epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
print(f'final loss: {loss.item():.4f}')
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}
FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#with open('intents', 'r') as f:
   # intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Kidjou3"
print("Let's chat! (type 'quit' to exit)")
while True:
    # sentence = "do you use credit cards?"
    sentence = input("You: ")
    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
     print(f"{bot_name}: I do not understand...")

def get_response(msg):
         sentence = tokenize(msg)
         X = bag_of_words(sentence, all_words)
         X = X.reshape(1, X.shape[0])
         X = torch.from_numpy(X).to(device)

         output = model(X)
         _, predicted = torch.max(output, dim=1)

         tag = tags[predicted.item()]

         probs = torch.softmax(output, dim=1)
         prob = probs[0][predicted.item()]
         if prob.item() > 0.75:
             for intent in intents:
                 if tag == intent["tag"]:
                     return random.choice(intent['responses'])

         return "I do not understand..."

if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(resp)
def index(request):
    return render(request, 'chatbotapp/index.html')

def chat(request):
    if request.method == 'POST':
        user_input = request.POST.get('user_input')
        bot_response = get_response(user_input)
        return JsonResponse({'bot_response': bot_response})
    else:
        return HttpResponseBadRequest('Méthode non autorisée')
