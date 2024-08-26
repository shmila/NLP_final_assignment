import os

from data_loading import load_all_transcriptions
from text_processing_utils import extract_valid_answers, extract_valid_image_descriptions

data_base_dir = r"C:\Users\elira\ShmilaJustSolveIt Dropbox\Eliran Shmila\PC\Documents\NLP\Final " \
                r"Project\data_from_beer_yaacov_2nd "
control_dir = os.path.join(data_base_dir, 'control', 'male')
patients_dir = os.path.join(data_base_dir, 'patients')

if __name__ == '__main__':
    # Load all files from control and patients directories
    control_transcriptions = load_all_transcriptions(control_dir)
    patient_transcriptions = load_all_transcriptions(patients_dir)

    print(f'Loaded {len(control_transcriptions)} control transcriptions.')
    print(f'Loaded {len(patient_transcriptions)} patient transcriptions.')

    # Extract valid answers

    # control_valid_answers = extract_valid_answers(control_transcriptions)
    # patient_valid_answers = extract_valid_answers(patient_transcriptions)
    #
    # print(sum([len(answer_list) for answer_list in control_valid_answers if len(answer_list) > 0]))
    # print(sum([len(answer_list) for answer_list in patient_valid_answers if len(answer_list) > 0]))

    control_valid_descriptions = extract_valid_image_descriptions(control_transcriptions)
    patient_valid_descriptions = extract_valid_image_descriptions(patient_transcriptions)
