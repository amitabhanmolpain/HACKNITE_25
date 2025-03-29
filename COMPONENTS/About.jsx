import { features } from '../constants';
import styles, { layout } from '../style';
import Button from './Button';

const FeatureCard = ({ icon, title, content, index }) => (
  <div className={`flex flex-row p-6 rounded-[20px] ${index !== features.length - 1 ? "mb-6" : "mb-0"} feature-card`}>
    <div className={`w-[64px] h-[64px] rounded-full ${styles.flexCenter} bg-dimBlue`}>
      <img src={icon} alt="icon" className="w-[50%] h-[50%] object-contain" />
    </div>
    <div className="flex-1 flex flex-col ml-3">
      <h4 className="font-poppins font-semibold text-white  text-[18px] leading-[23px] mb-1">
        {title}
      </h4>
      <p className="font-poppins font-normal  text-dimWhite  text-[16px] leading-[24px] mb-1">
        {content}
      </p>
    </div>
  </div>
)


const About = () => {
  return (
    <section id="features" className={layout.section}>
      <div className={layout.sectionInfo}>
        <h2 className={styles.heading2}>Empowering Education <br className="sm:block hidden" /> <span className="text-gradient">Through AI</span></h2>
        <p className={`${styles.paragraph} max-w-[470px] mt-5`}>
          ### About Siksha AI
          At Siksha AI, our mission is to revolutionize education by harnessing the power of advanced artificial intelligence technologies. We envision a future where every learner has access to personalized, engaging, and effective educational experiences that cater to their unique needs. Our core values—innovation, collaboration, and student-centered learning—guide us in our quest to create a more inclusive and accessible educational landscape. Our dedicated team, comprised of experts in education and technology, is passionate about transforming the way knowledge is delivered and absorbed. We take pride in the positive impact we've made, evidenced by numerous success stories and testimonials from students and educators who have benefited from our platform. Through strategic partnerships with educational institutions and organizations, we enhance our offerings and extend our reach. As we look to the future, we are excited to introduce new features and initiatives
          that will further empower learners and educators alike, ensuring that quality education is within everyone's reach.</p>

        <Button styles="mt-10" />
      </div>

      <div className={`${layout.sectionImg} flex-col`}>
        {features.map((feature, index) => (
          <FeatureCard key={feature.id} {...feature} index={index} />
        ))}
      </div>
    </section>
  )
}

export default About
