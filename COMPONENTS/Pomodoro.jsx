import React, { useState, useEffect } from 'react';
import { Play, Pause, RotateCcw } from 'lucide-react';

export default function Pomodoro() {
  const [minutes, setMinutes] = useState(25);
  const [seconds, setSeconds] = useState(0);
  const [isActive, setIsActive] = useState(false);
  const [selectedTime, setSelectedTime] = useState(25);

  useEffect(() => {
    let interval;

    if (isActive) {
      interval = setInterval(() => {
        if (seconds === 0) {
          if (minutes === 0) {
            setIsActive(false);
            return;
          }
          setMinutes(minutes - 1);
          setSeconds(59);
        } else {
          setSeconds(seconds - 1);
        }
      }, 1000);
    }

    return () => clearInterval(interval);
  }, [isActive, minutes, seconds]);

  const toggleTimer = () => {
    setIsActive(!isActive);
  };

  const resetTimer = () => {
    setIsActive(false);
    setMinutes(selectedTime);
    setSeconds(0);
  };

  const setTimer = (mins) => {
    setSelectedTime(mins);
    setMinutes(mins);
    setSeconds(0);
    setIsActive(false);
  };

  return (
    <div className="h-full flex flex-col">
      <div className="p-4 bg-gradient-dark border-b border-gray-800/50">
        <h2 className="text-xl font-semibold text-gray-100">Pomodoro Timer</h2>
      </div>

      <div className="flex-1 flex flex-col items-center justify-center p-6">
        <div className="flex gap-4 mb-8">
          <button
            onClick={() => setTimer(25)}
            className={`px-4 py-2 rounded-lg backdrop-blur-sm ${
              selectedTime === 25
                ? 'bg-blue-600/90 text-white'
                : 'bg-gradient-card border border-gray-800/50 text-gray-300'
            }`}
          >
            Pomodoro
          </button>
          <button
            onClick={() => setTimer(5)}
            className={`px-4 py-2 rounded-lg backdrop-blur-sm ${
              selectedTime === 5
                ? 'bg-blue-600/90 text-white'
                : 'bg-gradient-card border border-gray-800/50 text-gray-300'
            }`}
          >
            Short Break
          </button>
          <button
            onClick={() => setTimer(15)}
            className={`px-4 py-2 rounded-lg backdrop-blur-sm ${
              selectedTime === 15
                ? 'bg-blue-600/90 text-white'
                : 'bg-gradient-card border border-gray-800/50 text-gray-300'
            }`}
          >
            Long Break
          </button>
        </div>

        <div className="text-8xl font-bold mb-8 text-gray-100">
          {String(minutes).padStart(2, '0')}:
          {String(seconds).padStart(2, '0')}
        </div>

        <div className="flex gap-4">
          <button
            onClick={toggleTimer}
            className="p-4 bg-blue-600/90 text-white rounded-full hover:bg-blue-700/90 transition-colors backdrop-blur-sm"
          >
            {isActive ? <Pause size={24} /> : <Play size={24} />}
          </button>
          <button
            onClick={resetTimer}
            className="p-4 bg-gradient-card border border-gray-800/50 text-gray-300 rounded-full hover:bg-gray-800/50 transition-colors backdrop-blur-sm"
          >
            <RotateCcw size={24} />
          </button>
        </div>
      </div>
    </div>
  );
}