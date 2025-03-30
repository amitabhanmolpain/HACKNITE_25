import React, { useState, useEffect, useRef } from 'react';

const PomodoroClock = () => {
  const [timeLeft, setTimeLeft] = useState(25 * 60); // 25 minutes
  const [isRunning, setIsRunning] = useState(false);
  const [sessionType, setSessionType] = useState('Work');
  const [size, setSize] = useState(300); // Default size in pixels
  const timerRef = useRef(null);

  useEffect(() => {
    if (isRunning && timeLeft > 0) {
      timerRef.current = setInterval(() => {
        setTimeLeft(prevTime => prevTime - 1);
      }, 1000);
    } else if (timeLeft === 0) {
      clearInterval(timerRef.current);
      setSessionType(prev => (prev === 'Work' ? 'Break' : 'Work'));
      setTimeLeft(prev => (prev === 0 ? 5 * 60 : 25 * 60)); // Switch to break or reset to work
    }
    return () => clearInterval(timerRef.current);
  }, [isRunning, timeLeft]);

  const handleStartPause = () => {
    setIsRunning(prev => !prev);
  };

  const handleReset = () => {
    clearInterval(timerRef.current);
    setIsRunning(false);
    setTimeLeft(25 * 60);
    setSessionType('Work');
  };

  const formatTime = (seconds) => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
  };

  const handleSizeChange = (event) => {
    setSize(event.target.value);
  };

  return (
    <div className="flex flex-col items-center p-6 bg-gray-100 rounded-lg shadow-md max-w-md mx-auto">
      <h1 className="text-2xl font-bold mb-4 text-gray-800">{sessionType} Session</h1>
      <div 
        className="flex items-center justify-center" 
        style={{ width: `${size}px`, height: `${size}px`, borderRadius: '50%', border: '10px solid #4A90E2', position: 'relative' }}
      >
        <div className="flex items-center justify-center h-full">
          <div className="text-4xl font-bold">{formatTime(timeLeft)}</div>
        </div>
      </div>
      <div className="mt-4">
        <button 
          onClick={handleStartPause} 
          className="bg-blue-500 text-white py-2 px-4 rounded hover:bg-blue-600 transition duration-200 mb-2"
        >
          {isRunning ? 'Pause' : 'Start'}
        </button>
        <button 
          onClick={handleReset} 
          className="bg-red-500 text-white py-2 px-4 rounded hover:bg-red-600 transition duration-200 mb-2"
        >
          Reset
        </button>
      </div>
      <div className="mt-4">
        <label className="mr-2">Adjust Clock Size:</label>
        <input 
          type="range" 
          min="200" 
          max="600" 
          value={size} 
          onChange={handleSizeChange} 
          className="w-full"
        />
      </div>
    </div>
  );
};

export default PomodoroClock;