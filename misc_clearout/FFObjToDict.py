import wiplpy.WResults
import pickle


class FFObjToDictConverter:

    def __init__(self, ffobject, theta, frequency):
        self.ffobject = ffobject
        self.theta = theta
        self.frequency = frequency

        self.OUTPUT_GENERATED = False

    def extract_phi_results(self):
        re_Ephi = self.ffobject.GetYData(
            "Phi-component",
            "Re",
            XaxisLabel="phi",
            Cuts={"Theta": self.theta, "Frequency": self.frequency, "Excitation": 1},
        )
        im_Ephi = self.ffobject.GetYData(
            "Phi-component",
            "Im",
            XaxisLabel="phi",
            Cuts={"Theta": self.theta, "Frequency": self.frequency, "Excitation": 1},
        )

        return re_Ephi, im_Ephi

    def extract_theta_results(self):
        re_Etheta = self.ffobject.GetYData(
            "Theta-component",
            "Re",
            XaxisLabel="phi",
            Cuts={"Theta": self.theta, "Frequency": self.frequency, "Excitation": 1},
        )
        im_Etheta = self.ffobject.GetYData(
            "Theta-component",
            "Im",
            XaxisLabel="phi",
            Cuts={"Theta": self.theta, "Frequency": self.frequency, "Excitation": 1},
        )

        return re_Etheta, im_Etheta

    def extract_total_rcs(self):
        total_rcs = self.ffobject.GetYData(
            "Total",
            "RCS",
            XaxisLabel="phi",
            Cuts={"Theta": self.theta, "Frequency": self.frequency, "Excitation": 1},
        )

        return total_rcs

    def generate_output_dict(self):

        re_Ephi, im_Ephi = self.extract_phi_results()
        re_Etheta, im_Etheta = self.extract_theta_results()

        total_rcs = self.extract_total_rcs()

        self.output_dict = {
            "phi": self.ffobject.GetPhiPoints(),
            "Re_Ephi": re_Ephi,
            "Im_Ephi": im_Ephi,
            "Re_Etheta": re_Etheta,
            "Im_Etheta": im_Etheta,
            "Total_RCS": total_rcs,
        }

        self.output_dict["theta"] = self.theta
        self.output_dict["frequency"] = self.frequency

        self.OUTPUT_GENERATED = True

    def get_output_dict(self):

        if not self.OUTPUT_GENERATED:
            self.generate_output_dict()

        return self.output_dict

    def save_output_dict(self, file_path):

        if not self.OUTPUT_GENERATED:
            self.generate_output_dict()

        with open(file_path, "wb") as f:
            pickle.dump(self.output_dict, f)


if __name__ == "__main__":
    with open("X_V_alternate_run_results.pkl", "rb") as file:
        ffobject = pickle.load(file)

    theta = ffobject.GetThetaPoints()[0]
    frequency = ffobject.GetFrequencies()[0]

    results_extractor = FFObjToDictConverter(ffobject, theta, frequency)

    results_extractor.save_output_dict("X_V_alternate_run_results.pkl")
