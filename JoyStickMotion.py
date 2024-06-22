import keyboard
from time import sleep

buttons = {'x': 0, 'o': 0, 't': 0, 's': 0,
           'L1': 0, 'R1': 0, 'L2': 0, 'R2': 0,
           'share': 0, 'options': 0}

# Define the mapping from joystick buttons to keyboard keys
key_mapping = {
    'x': 'x',
    'o': 'o',
    't': 't',
    's': 's',
    'L1': 'a',
    'R1': 'd',
    'L2': 'q',
    'R2': 'e',
    'share': 'left shift',
    'options': 'right shift'
}

def getKeyboard(name=''):
    global buttons
    
    # Check each button
    for key, keyboard_key in key_mapping.items():
        if keyboard.is_pressed(keyboard_key):  # If the key is pressed
            buttons[key] = 1
        else:
            buttons[key] = 0

    if name == '':
        return buttons
    else:
        return buttons[name]

def main():
    print(getKeyboard('share'))  # To get a single value
    sleep(0.05)

if __name__ == '__main__':
    while True:
        main()