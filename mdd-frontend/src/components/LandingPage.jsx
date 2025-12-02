import React from "react";
import "../styles/LandingPage.css";

export default function LandingPage({ onStart }) {
  return (
    <div className="landing-container">

      {/* TOP MAIN HEADING */}
      <h1 className="main-heading">
  <div className="run-wrapper">
    <span className="running-text">Welcome to PronounceRight</span>
  </div>
</h1>



      {/* 2-COLUMN LAYOUT */}
      <div className="content-wrapper">

        {/* LEFT SIDE */}
        <div className="left-section">
          <h2 className="sub-heading">Words Matter</h2>
          <h2 className="sub-heading">Say It Right</h2>

          <p className="description">
            Get instant feedback, practice pronunciation, and follow your personalized daily study plan.
          </p>

          <button onClick={onStart} className="start-btn">
            Click here to start
          </button>
        </div>

        {/* RIGHT SIDE */}
        <div className="right-section">
          <img src="/1.jpg" alt="Two people speaking" className="hero-image" />
        </div>

      </div>
    </div>
  );
}
