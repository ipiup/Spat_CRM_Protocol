import random
import pandas as pd

# Paramètres
soundpath = 'C:\\Users\\chamery\\Documents\\Spat_CRM_Protocol\\Spat_CRM_Protocol\\sounds'
outputpath = 'C:\\Users\\chamery\\Documents\\Spat_CRM_Protocol\\Spat_CRM_Protocol\\src\\conditions'
talkers = ['T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8']
genders = ['M', 'F']
mask_callsigns = ['Alpha', 'Charlie', 'Echo', 'Kilo', 'Oscar', 'Tango', 'Whisky']
target_callsign = 'Delta'
colors = ['bleu', 'jaune', 'rouge', 'vert']
digits = ['1', '2', '3', '4', '5', '6', '7', '8']
mask_angles = ['10', '20', '30', '40', '50', '60', '70', '80', '90']
target_angle = '0'
num_trials = 32

def generate_filename(talker, gender, callsign, color, digit, angle):
    return f"{talker}_{gender}_{callsign}_{color}_{digit}_{angle}.wav"

def generate_valid_pair():
    while True:
        # random target (delta at 0°)
        target_talker = random.choice(talkers)
        target_gender = random.choice(genders)
        target_color = random.choice(colors)
        target_digit = random.choice(digits)
        target = generate_filename(
            target_talker, target_gender, target_callsign,
            target_color, target_digit, target_angle
        )

        # Mask (diff gender, diff talker)
        mask_gender = 'F' if target_gender == 'M' else 'M'
        mask_talker = random.choice([t for t in talkers if t != target_talker])
        mask_callsign = random.choice(mask_callsigns)
        mask_angle = random.choice(mask_angles)

        # mask and talker different color digit combinaison
        possible_dc = [(d, c) for d in digits for c in colors if (d, c) != (target_digit, target_color)]
        mask_digit, mask_color = random.choice(possible_dc)

        mask = generate_filename(
            mask_talker, mask_gender, mask_callsign,
            mask_color, mask_digit, mask_angle
        )
        return {
            "Target": target,
            "Target_Talker": target_talker,
            "Target_Gender": target_gender,
            "Target_Callsign": target_callsign,
            "Target_Color": target_color,
            "Target_Digit": target_digit,
            "Target_Angle": target_angle,
            "Mask": mask,
            "Mask_Talker": mask_talker,
            "Mask_Gender": mask_gender,
            "Mask_Callsign": mask_callsign,
            "Mask_Color": mask_color,
            "Mask_Digit": mask_digit,
            "Mask_Angle": mask_angle
        }

def generate_csv(num_trials, output_file=f"{outputpath}\\conditions_2loc.csv"):
    trials = []

    while len(trials) < num_trials:
        pair_data = generate_valid_pair()
        if not any(
            (t["Target"] == pair_data["Target"] and t["Mask"] == pair_data["Mask"])
            for t in trials
        ):
            trials.append(pair_data)

    df = pd.DataFrame(trials)
    columns_order = [
        "Target", "Target_Talker", "Target_Gender", "Target_Callsign",
        "Target_Color", "Target_Digit", "Target_Angle",
        "Mask", "Mask_Talker", "Mask_Gender", "Mask_Callsign",
        "Mask_Color", "Mask_Digit", "Mask_Angle"
    ]
    df = df[columns_order]

    df.to_csv(output_file, index=False)
    print(f"Fichier généré : {output_file}")

if __name__ == "__main__":
    generate_csv(num_trials=32)