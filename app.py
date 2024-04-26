import streamlit as st
import numpy as np
from PIL import Image
import io
import rembg
import tensorflow as tf
from tensorflow.keras.models import Sequential, model_from_json
import json
import cv2


def capture_image():
    """
    Capture image from webcam.
    """
    return st.sidebar.camera_input("Take a picture")

def remove_and_replace_background(image):
    """
    Remove background from the image and replace it with white.
    """
    # img_array = np.array(image)
    output_array = rembg.remove(image)
    background = Image.new("RGB", (image.shape[1], image.shape[0]), (255, 255, 255))  # Pass size explicitly
    output_image = Image.fromarray(output_array)
    background.paste(output_image, (0, 0), output_image)
    return np.array(background)


def preprocess_image(picture):
    """
    Preprocesses the image to meet the required format.
    """
    # Read image bytes from the UploadedFile object
    image_bytes = picture.read()
    
    # Convert image bytes to a PIL Image object
    img = Image.open(io.BytesIO(image_bytes))
    
    # Convert PIL Image to NumPy array
    image_array = np.array(img)

    image_array = remove_and_replace_background(image_array)

    # Resize the image to (180, 180) if necessary using OpenCV
    if image_array.shape[:2] != (180, 180):
        image_array = cv2.resize(image_array, (180, 180))

    # Ensure the image has 3 channels
    if len(image_array.shape) == 2:
        image_array = np.stack((image_array,) * 3, axis=-1)

    # Remove singleton dimensions
    image_array = np.squeeze(image_array)

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    return image_array.astype(np.float32)
def preprocess_image(picture):
    """
    Preprocesses the image to meet the required format.
    """
    # Read image bytes from the UploadedFile object
    image_bytes = picture.read()
    
    # Convert image bytes to a PIL Image object
    img = Image.open(io.BytesIO(image_bytes))
    
    # Convert PIL Image to NumPy array
    image_array = np.array(img)

    image_array = remove_and_replace_background(image_array)

    # Resize the image to (180, 180) if necessary using OpenCV
    if image_array.shape[:2] != (180, 180):
        image_array = cv2.resize(image_array, (180, 180))

    # Ensure the image has 3 channels
    if len(image_array.shape) == 2:
        image_array = np.stack((image_array,) * 3, axis=-1)

    # Reshape the image array
    image_array = image_array.reshape((180, 180, 3))

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    return image_array.astype(np.float32)


def computer_choice():
    """
    Randomly selects the computer's choice: rock, paper, or scissor.
    """
    choice = np.random.randint(1, 4)
    if choice == 1:
        return "Rock"
    elif choice == 2:
        return "Paper"
    else:
        return "Scissor"

def load_model_from_disk():
    # Load the model architecture from JSON file
    with open('saved_model.json', 'r') as json_file:
        model_config = json_file.read()

    # Load model architecture from JSON string
    loaded_model = tf.keras.models.model_from_json(model_config)

    # Load weights into the model
    loaded_model.load_weights("saved_model.h5")
    # print("Loaded model architecture and weights from disk")

    # Compile the model
    loaded_model.compile(optimizer='adam',
                         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                         metrics=['accuracy'])

    return loaded_model

def make_prediction(processed_image, loaded_model):

    class_names = ['Paper', 'Rock', 'Scissor']

    predictions = loaded_model.predict(processed_image)
    score = tf.nn.softmax(predictions[0])
    predicted_label = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    return predicted_label, confidence

def rock_paper_scissors(player_choice, computer_choice):
    """
    Simulates a round of Rock Paper Scissors game.
    
    Parameters:
        player_choice (str): The player's choice (either 'rock', 'paper', or 'scissors').
        computer_choice (str): The computer's choice (either 'rock', 'paper', or 'scissors').
    
    Returns:
        str: The result of the game ('Player wins', 'Computer wins', or 'It's a tie').
    """
    player_choice = player_choice.lower()
    computer_choice = computer_choice.lower()

    if player_choice == computer_choice:
        return "It's a tie"
    elif (player_choice == 'rock' and computer_choice == 'scissor') or \
         (player_choice == 'paper' and computer_choice == 'rock') or \
         (player_choice == 'scissor' and computer_choice == 'paper'):
        return "Player wins"
    else:
        return "Computer wins"

def get_icon(choice):
    """
    Returns an icon based on the choice (rock, paper, or scissors).
    """
    if choice == 'rock':
        return "🪨"
    elif choice == 'paper':
        return "📄"
    elif choice == 'scissor':
        return "✂️"
    else:
        return ""

def main():
    st.set_page_config(page_title="Rock Paper Scissor Game", layout="wide")

    loaded_model = load_model_from_disk()
    
    st.sidebar.title("Webcam Image Capture")
    picture = capture_image()
    if picture:
        st.sidebar.write("Captured Image:")
        st.sidebar.image(picture, use_column_width=True)

    st.title("Rock Paper Scissor Game")
    st.markdown("---") 
    container = st.container()
    with container:
        if picture:
            processed_image = preprocess_image(picture)

            pred_label, confidence_score = make_prediction(processed_image, loaded_model)   

            # Choose color based on prediction confidence level
            confidence_color = "lightgreen" if confidence_score > 90 else "green" if confidence_score > 70 else "orange" if confidence_score > 50 else "red"

            # Display prediction confidence with different colors
            st.markdown(f"<h2 style='text-align: left; color: {confidence_color};'>Prediction Confidence: {confidence_score:.2f}%</h2>", unsafe_allow_html=True)
            
            computer_choice_text = computer_choice()

            # Play Rock Paper Scissors game
            result = rock_paper_scissors(pred_label.lower(), computer_choice_text.lower())

            # Get icons for player and computer choices
            player_choice_icon = get_icon(pred_label.lower())
            computer_choice_icon = get_icon(computer_choice_text.lower())

            # Display player and computer choices with icons and text
            st.markdown(f"<h2 style='text-align: left; color: green;'>Your Choice: {player_choice_icon} {pred_label}</h2>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='text-align: left; color: red;'>Computer's Choice: {computer_choice_icon} {computer_choice_text}</h2>", unsafe_allow_html=True)
            
            # Display result with corresponding color and emojis
            if result == "Player wins":
                winner_emoji = "👤"  # Human emoji
                result_color = "green"
            elif result == "Computer wins":
                winner_emoji = "🤖"  # Computer emoji
                result_color = "red"
            else:
                winner_emoji = ""  # No winner
                result_color = "white"

            st.markdown(f"<h2 style='text-align: left; color: {result_color};'>Result: {result} {winner_emoji}</h2>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
