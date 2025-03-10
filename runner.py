import wiplpy.WiplInterface
import wiplpy.WResults
from FFObjToDict import FFObjToDictConverter

PROJECT_PATH = r"E:\Working_Copy\Bernard_Ellipsoid_Comparison\Bernard\LepidopteraNoctuidaeMoth_12_Full_M_0250_0375_94_MirkLBG_ChenMeanandLegs"
# PROJECT_PATH = r"C:\Users\NCAS\Documents\Tommy\purple_martin_work\P_Martin2\X_band\horizontal_alternate_run\M_Wing_1_100_H_X"
# PROJECT_PATH = r"C:\Users\NCAS\Documents\Tommy\purple_martin_work\P_Martin2\S_band\horizontal_alternate_run\M_Wing_1_100_H_alternate_run"

WIPLDInstallDirectory = r"C:\WIPL-D Pro CAD 2024"

# BASE_FILE_NAME = "S_H_alternate_run"
BASE_FILE_NAME = "FullSize_B_freq_sweep_H_1_60_20"
SAVE_PATH = r"C:\Users\NCAS\Documents\Tommy\Bernard_run_outputs\\horizontal\\"

pro = wiplpy.WiplInterface.InitializeWIPLDSuite(WIPLDInstallDirectory, "wipldpro")

pro.Run(PROJECT_PATH)

far_field = wiplpy.WResults.InitializeFFResults(PROJECT_PATH)

theta = far_field.GetThetaPoints()[0]
frequencies = far_field.GetFrequencies()

if len(frequencies) > 1:
    print("Multiple frequencies present")
    for counter, frequency in enumerate(far_field.GetFrequencies()):
        print(f"Saving frequency number {counter}, value {frequency} GHz")
        results_extractor = FFObjToDictConverter(far_field, theta, frequency)
        results_extractor.save_output_dict(
            SAVE_PATH + f"{BASE_FILE_NAME}_f{counter}_dict.pkl"
        )

else:
    print("Only one frequency present")
    frequency = frequencies[0]
    results_extractor = FFObjToDictConverter(far_field, theta, frequency)
    results_extractor.save_output_dict(SAVE_PATH + f"{BASE_FILE_NAME}_dict.pkl")
