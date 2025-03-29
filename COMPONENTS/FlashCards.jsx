import React, { useState } from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';

export default function FlashCards() {
  const [isFlipped, setIsFlipped] = useState(false);
  const [currentIndex, setCurrentIndex] = useState(0);

  // Example flashcards
  const flashcards = [
    {
      question: "What is React?",
      answer: "A JavaScript library for building user interfaces"
    },
    // Add more flashcards here
  ];

  const handleNext = () => {
    setIsFlipped(false);
    setCurrentIndex((prev) => (prev + 1) % flashcards.length);
  };

  const handlePrevious = () => {
    setIsFlipped(false);
    setCurrentIndex((prev) => (prev - 1 + flashcards.length) % flashcards.length);
  };

  return (
    <div className="h-full flex flex-col">
      <div className="p-4 bg-gradient-dark border-b border-gray-800/50">
        <h2 className="text-xl font-semibold text-gray-100">Flash Cards</h2>
      </div>

      <div className="flex-1 p-6 flex flex-col items-center justify-center">
        <div className="relative w-full max-w-2xl aspect-[3/2]">
          <div
            className={`w-full h-full transition-all duration-500 ${
              isFlipped ? '[transform:rotateY(180deg)]' : ''
            }`}
            style={{ transformStyle: 'preserve-3d' }}
          >
            {/* Front of card */}
            <div
              onClick={() => setIsFlipped(true)}
              className="absolute w-full h-full bg-gradient-card border border-gray-800/50 rounded-xl p-8 flex items-center justify-center cursor-pointer backdrop-blur-sm"
              style={{ backfaceVisibility: 'hidden' }}
            >
              <p className="text-2xl text-center text-gray-100">
                {flashcards[currentIndex]?.question}
              </p>
            </div>

            {/* Back of card */}
            <div
              onClick={() => setIsFlipped(false)}
              className="absolute w-full h-full bg-gradient-card border border-gray-700/50 rounded-xl p-8 flex items-center justify-center cursor-pointer backdrop-blur-sm [transform:rotateY(180deg)]"
              style={{ backfaceVisibility: 'hidden' }}
            >
              <p className="text-2xl text-center text-gray-100">
                {flashcards[currentIndex]?.answer}
              </p>
            </div>
          </div>
        </div>

        <div className="mt-8 flex items-center gap-4">
          <button
            onClick={handlePrevious}
            className="p-2 bg-gradient-card border border-gray-800/50 text-gray-300 rounded-full hover:bg-gray-800/50 backdrop-blur-sm"
          >
            <ChevronLeft size={24} />
          </button>
          <span className="text-lg text-gray-300">
            {currentIndex + 1} / {flashcards.length}
          </span>
          <button
            onClick={handleNext}
            className="p-2 bg-gradient-card border border-gray-800/50 text-gray-300 rounded-full hover:bg-gray-800/50 backdrop-blur-sm"
          >
            <ChevronRight size={24} />
          </button>
        </div>
      </div>
    </div>
  );
}