import styles from '../style';

import GetStarted from './GetStarted';

const Hero = () => {
  return (
    <section id="home" className={`flex md:flex-row flex-col ${styles.paddingY}`}>
      <div className={`flex-1 ${styles.flexStart} flex-col xl:px-0 sm:px-16 px-6`}>
        

        <div className="flex flex-row justify-between items-center w-full">
          <h1 className="flex-1 font-poppins font-semibold ss:text-[72px] text-[52px] text-white ss:leading-[100px] leading-[75px] ">
          Where Curiosity Meets Knowledge:<br className="sm:block hidden"/>{" "}
            <span className="text-gradient">Ignite Your Learning Journey</span>{" "}
          </h1>
          
        </div>

        <h1 className="font-poppins font-semibold ss:text-[68px] text-[52px] text-white ss:leading-[100px] leading-[75px] w-full">
        </h1>

        <p className={`${styles.paragraph} text-dimWhite max-w-[470px] mt-5`}>
        At ShiksaAI, we believe that education is the key to unlocking a world of possibilities. Our platform is dedicated to empowering learners of all ages with the knowledge and skills they need to thrive in an ever-changing world. Whether you're a student seeking to enhance your understanding of complex subjects, a professional looking to upskill, or a lifelong learner eager to explore new interests, we have something for everyone.
        </p>
      </div>

      <div className={`flex-1 flex ${styles.flexCenter} md:my-0 my-10 relative`}>
        

        <div className="absolute z-[0] w-[40%] h-[35%] top-0 pink__gradient"/>
        <div className="absolute z-[1] w-[90%] h-[90%] rounded-full bottom-40 white__gradient"/>
        <div className="absolute z-[0] w-[50%] h-[50%] right-20 bottom-20 blue__gradient"/>
      </div>

      <div className={`ss:hidden ${styles.flexCenter}`}>
      </div>
    </section>
  );
};

export default Hero;