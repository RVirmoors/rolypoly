# HELPER METHODS

def get_y_n(prompt):
    while True:
        try:
            return {"y" or " ": True, "n": False}[input(prompt).lower()]
        except KeyError:
            print("Invalid input---please enter Y or N!")