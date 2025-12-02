import { useState } from "react";
import LandingPage from "./components/LandingPage";
import RecordingPage from "./components/RecordingPage";

function App() {
  const [step, setStep] = useState("landing");

  const startRecordingFlow = () => {
    setStep("recording");
  };

  const goBackToLanding = () => {
    setStep("landing");
  };

  return (
     <>
      {step === "landing" && (
        <LandingPage onStart={startRecordingFlow} />
      )}

      {step === "recording" && (
        <RecordingPage onBack={goBackToLanding} />
      )}
    </>
  );
}

export default App;
