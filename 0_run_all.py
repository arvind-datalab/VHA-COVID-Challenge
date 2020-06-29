import os
import subprocess

subprocess.call('python 1_patient_finding.py', shell=True)
subprocess.call('python 2_demographics.py', shell=True)
subprocess.call('python 3_Condition_Train_Disorder.py', shell=True)
subprocess.call('python 4_Condition_Train_Finding.py', shell=True)
subprocess.call('python 5_medication.py', shell=True)
subprocess.call('python 6_immunization.py', shell=True)
subprocess.call('python 7_observation.py', shell=True)
subprocess.call('python 8_procedure.py', shell=True)
subprocess.call('python 9_careplan.py', shell=True)
subprocess.call('python 10_devices.py', shell=True)
subprocess.call('python 11_encounters.py', shell=True)
subprocess.call('python 12_image_allergies_code.py', shell=True)
subprocess.call('python 13_payers.py', shell=True)
subprocess.call('python 14_car_record_feature_selection.py', shell=True)
subprocess.call('python 15_covid_vs_no_covid_gbm_model.py', shell=True)