import wiplpy.WiplInterface
import wiplpy.WResults
import pickle

PROJECT_PATH = r"C:\Users\NCAS\Documents\Tommy\purple_martin_work\P_Martin2\X_band\vertical_alternate_run_test_dummy\M_Wing_1_100_V_X_alternate_run"

# WIPLDInstallDirectory = r"C:\Program Files\WIPL-D\WIPL-D Pro 13.02.00"

# WIPLD = wiplpy.WiplInterface(WIPLDInstallDirectory, "wipldpro")

far_field = wiplpy.WResults.InitializeFFResults(PROJECT_PATH)

theta = far_field.GetThetaPoints()[0]
frequency = far_field.GetFrequencies()[0]

print(theta, frequency)
phi_results = far_field.GetPhiPoints()

# print(phi_results)

results_by_phi = far_field.GetYData(
    "Phi-component",
    "Re",
    XaxisLabel="phi",
    Cuts={"Theta": theta, "Frequency": frequency, "Excitation": 1},
)

# Pickle the far_field object
with open("X_V_alternate_run_results.pkl", "wb") as f:
    pickle.dump(far_field, f)

print(results_by_phi)
# think I need to try this on a run that has been run using a built in set of excitations rather than a sweep.
