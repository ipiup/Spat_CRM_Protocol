import random
import pandas as pd
from typing import Dict, List, Tuple

# Paramètres
soundpath = 'E:\\ACERI\\CRMFRvsEN\\02SCRIPTS\\Tache\\sound\\FR'
outputpath = 'C:\\Users\\chamery\\Documents\\Spat_CRM_Protocol\\Spat_CRM_Protocol\\src\\conditions'

# Talkers avec genre fixe : T0-T3 = M, T4-T7 = F
talkers = ['T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7']
talker_genders = {'T0': 'M', 'T1': 'M', 'T2': 'M', 'T3': 'M', 'T4': 'F', 'T5': 'F', 'T6': 'F', 'T7': 'F'}
mask_callsigns = ['Alpha', 'Charlie', 'Echo', 'Kilo', 'Oscar', 'Tango', 'Whisky']
target_callsign = 'Delta'
colors = ['bleu', 'jaune', 'rouge', 'vert']
digits = ['1', '2', '3', '4', '5', '6', '7', '8']
mask_angles = ['10', '20', '30', '40', '50', '60', '70', '80', '90']
target_angle = '0'
num_trials = 10

# Fonctions
def generate_filename(talker: str, gender: str, callsign: str, color: str, digit: str, angle: str) -> str:
    #return f"{talker}_{gender}_{callsign}_{color}_{digit}_{angle}.wav"
    return f"{talker}_{gender}_{callsign}_{color}_{digit}.wav"

def save_to_csv(data: List[Dict], output_file: str, columns_order: List[str]) -> None:
    df = pd.DataFrame(data)
    df = df[columns_order]
    df.to_csv(output_file, index=False)
    print(f"Fichier généré : {output_file}")

# ----- 2 LOC (1 target + 1 masker)
def generate_1mask_trial() -> Dict:
    # Target (genre déterminé par le talker)
    target_talker = random.choice(talkers)
    target_gender = talker_genders[target_talker]
    target_color = random.choice(colors)
    target_digit = random.choice(digits)
    target = generate_filename(
        target_talker, target_gender, target_callsign,
        target_color, target_digit, target_angle
    )

    # Mask (genre opposé, talker différent)
    mask_gender = 'F' if target_gender == 'F' else 'M'
    # Filtrer les talkers avec le bon genre
    possible_mask_talkers = [t for t in talkers if talker_genders[t] == mask_gender and t != target_talker]
    mask_talker = random.choice(possible_mask_talkers)
    mask_callsign = random.choice(mask_callsigns)
    mask_angle = random.choice(mask_angles)

    # digit+couleur Masker != Target
    possible_dc = [(d, c) for d in digits for c in colors if (d, c) != (target_digit, target_color)]
    mask_digit, mask_color = random.choice(possible_dc)
    mask = generate_filename(
        mask_talker, mask_gender, mask_callsign,
        mask_color, mask_digit, mask_angle
    )

    return {
        "Target_2loc": f'{soundpath}\\{target}', "Target_Talker_2loc": target_talker, "Target_Gender_2loc": target_gender,
        "Target_Callsign_2loc": target_callsign, "Target_Color_2loc": target_color,
        "Target_Digit_2loc": target_digit, "Target_Angle_2loc": target_angle,
        "Mask_2loc": f'{soundpath}\\{mask}', "Mask_Talker_2loc": mask_talker, "Mask_Gender_2loc": mask_gender,
        "Mask_Callsign_2loc": mask_callsign, "Mask_Color_2loc": mask_color,
        "Mask_Digit_2loc": mask_digit, "Mask_Angle_2loc": mask_angle
    }

def generate_1mask_csv() -> None:
    trials = []
    columns_order = [
        "Target_2loc", "Target_Talker_2loc", "Target_Gender_2loc", "Target_Callsign_2loc", "Target_Color_2loc", "Target_Digit_2loc", "Target_Angle_2loc",
        "Mask_2loc", "Mask_Talker_2loc", "Mask_Gender_2loc", "Mask_Callsign_2loc", "Mask_Color_2loc", "Mask_Digit_2loc", "Mask_Angle_2loc"
    ]

    while len(trials) < num_trials:
        trial = generate_1mask_trial()
        # Éviter les doublons
        if not any(t["Target_2loc"] == trial["Target_2loc"] and t["Mask_2loc"] == trial["Mask_2loc"] for t in trials):
            trials.append(trial)

    save_to_csv(trials, f"{outputpath}\\conditions_2loc.csv", columns_order)

# --------- 3 LOC (1 target + 2 maskers)
def generate_2maskers_trial() -> Dict:
    # Target (genre déterminé par le talker)
    target_talker = random.choice(talkers)
    target_gender = talker_genders[target_talker]
    target_color = random.choice(colors)
    target_digit = random.choice(digits)
    target = generate_filename(
        target_talker, target_gender, target_callsign,
        target_color, target_digit, target_angle
    )

    # Angles opposés pour les Maskers (ex: +30 et -30)
    base_angle = random.choice(mask_angles)
    mask_angle1 = base_angle
    mask_angle2 = f"-{base_angle}"

    # Mask1 (même genre que Target, talker différent)
    possible_mask_talkers = [t for t in talkers if talker_genders[t] == target_gender and t != target_talker]
    mask1_talker = random.choice(possible_mask_talkers)
    mask1_callsign = random.choice(mask_callsigns)
    possible_dc1 = [(d, c) for d in digits for c in colors if (d, c) != (target_digit, target_color)]
    mask1_digit, mask1_color = random.choice(possible_dc1)
    mask1 = generate_filename(
        mask1_talker, target_gender, mask1_callsign,
        mask1_color, mask1_digit, mask_angle1
    )

    # Mask2 (même genre, talker différent de Target et Mask1, digit+couleur différent)
    possible_mask2_talkers = [t for t in talkers if talker_genders[t] == target_gender and t != target_talker and t != mask1_talker]
    mask2_talker = random.choice(possible_mask2_talkers)
    mask2_callsign = random.choice(mask_callsigns)
    possible_dc2 = [
        (d, c) for d in digits for c in colors
        if (d, c) != (target_digit, target_color) and (d, c) != (mask1_digit, mask1_color)
    ]
    mask2_digit, mask2_color = random.choice(possible_dc2)
    mask2 = generate_filename(
        mask2_talker, target_gender, mask2_callsign,
        mask2_color, mask2_digit, mask_angle2
    )

    return {
        "Target_3loc": f'{soundpath}\\{target}', "Target_Talker_3loc": target_talker, "Target_Gender_3loc": target_gender,
        "Target_Callsign_3loc": target_callsign, "Target_Color_3loc": target_color,
        "Target_Digit_3loc": target_digit, "Target_Angle_3loc": target_angle,
        "Mask1_3loc": f'{soundpath}\\{mask1}', "Mask1_Talker_3loc": mask1_talker, "Mask1_Gender_3loc": target_gender,
        "Mask1_Callsign_3loc": mask1_callsign, "Mask1_Color_3loc": mask1_color,
        "Mask1_Digit_3loc": mask1_digit, "Mask1_Angle_3loc": mask_angle1,
        "Mask2_3loc": f'{soundpath}\\{mask2}', "Mask2_Talker_3loc": mask2_talker, "Mask2_Gender_3loc": target_gender,
        "Mask2_Callsign_3loc": mask2_callsign, "Mask2_Color_3loc": mask2_color,
        "Mask2_Digit_3loc": mask2_digit, "Mask2_Angle_3loc": mask_angle2
    }

def generate_2maskers_csv() -> None:
    trials = []
    columns_order = [
        "Target_3loc", "Target_Talker_3loc", "Target_Gender_3loc", "Target_Callsign_3loc", "Target_Color_3loc", "Target_Digit_3loc", "Target_Angle_3loc",
        "Mask1_3loc", "Mask1_Talker_3loc", "Mask1_Gender_3loc", "Mask1_Callsign_3loc", "Mask1_Color_3loc", "Mask1_Digit_3loc", "Mask1_Angle_3loc",
        "Mask2_3loc", "Mask2_Talker_3loc", "Mask2_Gender_3loc", "Mask2_Callsign_3loc", "Mask2_Color_3loc", "Mask2_Digit_3loc", "Mask2_Angle_3loc"
    ]

    while len(trials) < num_trials:
        trial = generate_2maskers_trial()
        # Éviter les doublons
        if not any(
            t["Target_3loc"] == trial["Target_3loc"] and
            t["Mask1_3loc"] == trial["Mask1_3loc"] and
            t["Mask2_3loc"] == trial["Mask2_3loc"]
            for t in trials
        ):
            trials.append(trial)

    save_to_csv(trials, f"{outputpath}\\conditions_3loc.csv", columns_order)

if __name__ == "__main__":
    generate_1mask_csv()      # Génère conditions_2loc.csv
    generate_2maskers_csv()  # Génère conditions_3loc.csv