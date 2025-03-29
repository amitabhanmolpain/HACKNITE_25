import styles from './style';
import { About,  Services, Clients, CTA, Footer, Navbar, Hero, Chatbot } from "./components";



const App = () => (
  <div className="bg-primary w-full overflow-hidden">
    <div className={`${styles.paddingX} ${styles.flexCenter}`}>
      <div className={`${styles.boxWidth}`}>
        <Navbar />
      </div>
    </div>

    <div className={`bg-primary ${styles.flexStart}`}>
      <div className={`${styles.boxWidth}`}>
        <Hero />
      </div>
    </div>

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

    {/* floating Chatbot */}
    <Chatbot />
  </div>
);

export default App;