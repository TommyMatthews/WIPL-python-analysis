import wiplpy.WiplInterface
import wiplpy.WResults
import wiplpy.WSymbols as symb
from FFObjToDict import FFObjToDictConverter

PROJECT_PATH = r"E:\Working_Copy\Bernard_Ellipsoid_Comparison\Ellipsoid\ellipsoid"
SYMB_PATH = r"E:\Working_Copy\Bernard_Ellipsoid_Comparison\Ellipsoid\ellipsoid.SMB"
# PROJECT_PATH = r"C:\Users\NCAS\Documents\Tommy\purple_martin_work\P_Martin2\X_band\horizontal_alternate_run\M_Wing_1_100_H_X"
# PROJECT_PATH = r"C:\Users\NCAS\Documents\Tommy\purple_martin_work\P_Martin2\S_band\horizontal_alternate_run\M_Wing_1_100_H_alternate_run"

WIPLDInstallDirectory = r"C:\WIPL-D Pro CAD 2024"

run_list = {
    "H" : [1,0],
    "V" : [0,1]
}

pro = wiplpy.WiplInterface.InitializeWIPLDSuite(WIPLDInstallDirectory, "wipldpro")

for run, values in run_list.items():

    # SymbolsList =symb.GetSymbols(SYMB_PATH)
    # SymbolsList.SetSymbolByName("Ephi", values[0])
    # SymbolsList.SetSymbolByName("Etheta", values[1])
    # Read the content of the file
    with open(SYMB_PATH, 'r') as file:
        content = file.readlines()

    # Modify the content with new values
    for i, line in enumerate(content):
        if 'Ephi' in line:
            content[i] = f"Ephi={values[0]}\n"
        elif 'Etheta' in line:
            content[i] = f"Etheta={values[1]}\n"

    # Write the updated content back to the file
    with open(SYMB_PATH, 'w') as file:
        file.writelines(content)
    print(f"Running {run} run")
    print(content)

    BASE_FILE_NAME = f"Ellipsoid_0375_sweep_{run}_1_60_20"
    SAVE_PATH = f"C:\\Users\\NCAS\\Documents\\Tommy\\Ellipsoid_run_outputs\\0375\\{run}\\"

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
