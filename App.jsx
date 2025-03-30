import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import styles from './style';
import { 
  About, 
  Services, 
  Clients, 
  CTA, 
  Footer, 
  Navbar, 
  Hero, 
  Chatbot 
} from "./components"; // Import components from the components folder
import PdfTools from './pages/PdfTools'; // Updated import path
import Pomodoro from './pages/Pomodoro'; // Updated import path
import FlashCards from './pages/FlashCards'; // Updated import path
import chatinte from './pages/chatinte'; // Updated import path

const App = () => {
  const navigateToPage = (path) => {
    window.location.href = path; // Change the window location to the new path
  };

  return (
    <Router>
      <div className="bg-primary w-full overflow-hidden">
        {/* Navbar Section */}
        <div className={`${styles.paddingX} ${styles.flexCenter}`}>
          <div className={`${styles.boxWidth}`}>
            <Navbar />
          </div>
        </div>

        {/* Hero Section */}
        <div className={`bg-primary ${styles.flexStart}`}>
          <div className={`${styles.boxWidth}`}>
            <Hero />
          </div>
        </div>

        {/* Main Content Section */}
        <div className={`bg-primary ${styles.paddingX} ${styles.flexCenter}`}>
          <div className={`${styles.boxWidth}`}>
            <div id="About">
              <About />
            </div>
            <div id="Services">
              <Services />
            </div>
            <div id="Clients">
              <Clients />
            </div>
            <CTA />
            <Footer />
          </div>
        </div>

        {/* Floating Chatbot */}
        <Chatbot />
      </div>

      {/* Define routes for your application */}
      <Routes>
        <Route path="/" element={<Hero />} /> {/* Home Route */}
        <Route path="/pdf-tools" element={<PdfTools />} />
        <Route path="/pomodoro" element={<Pomodoro />} />
        <Route path="/flashcards" element={<FlashCards />} />
        <Route path="/chat-inte" element={<chatinte />} />
        {/* Add other routes as needed */}
      </Routes>

      {/* Navigation Buttons */}
      <div className="fixed bottom-4 right-4">
        <button 
          onClick={() => navigateToPage('/pdf-tools')} 
        >
          PDF Summarizer
        </button>
        <button 
          onClick={() => navigateToPage('/pomodoro')} 
        >
          Pomodoro
        </button>
        <button 
          onClick={() => navigateToPage('/flashcards')} 
        >
          Flashcards
        </button>
      </div>
    </Router>
  );
};

export default App;
