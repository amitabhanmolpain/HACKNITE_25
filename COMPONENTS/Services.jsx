import useNavigate from 'react';
import { services } from '../constants'; // Import services data
import styles from '../style'; // Import styles


// ServiceCard Component
const ServiceCard = ({ icon, title, content }) => (
  <div className="flex flex-col items-center p-6 bg-black-gradient   rounded-[20px] shadow-md transform transition duration-300 hover:scale-105">
    <img src={icon} alt={title} className="w-[64px] h-[64px] mb-4 object-contain" />
    <h3 className="font-poppins font-semibold text-[20px] leading-[32px] text-white mb-2">{title}</h3>
    <p className="font-poppins font-normal text-[16px] leading-[24px] text-dimWhite text-center">{content}</p>
  </div>
);

// Services Component
function Services() {
  const navigate = useNavigate()
  return (
    <section id="services" className={`${styles.paddingY} flex-col relative`}>
      <div className="w-full flex justify-center items-center flex-col mb-10">
        <h2 className="font-poppins font-semibold text-[50px] text-white leading-[66.8px] w-full text-center">
          Our <span class="text-gradient">Services</span> 
        </h2>
        <p className="font-poppins font-normal text-[18px] leading-[30.8px] text-dimWhite text-center max-w-[600px]">
          Discover the range of services we offer to make Learning accessible and convenient for you.
        </p>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 w-full">
        {services.map((service) => (
          <ServiceCard key={service.id} {...service} />
        ))}
      </div>
    </section>
  )
};

export default Services;