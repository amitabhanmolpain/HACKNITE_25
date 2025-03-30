import {  facebook, instagram, linkedin, twitter, airbnb, binance, coinbase, dropbox, testtube,inject,medicine} from "../assets";
import { insurance, helpline,appointments,chatbot,healthrecord,meddeli ,robot,pdf,educan,clock,staistics,globe,tw7,flashCards,buddy} from "../assets";


export const navLinks = [
  {
    id: "home",
    title: "Home",
  },
  {
    id: "About",
    title: "About",
  },
  {
    id: "Services",
    title: "Services",
  },
  {
    id: "Feedbacks",
    title: "Feedbacks",
  },
];

export const features = [
  {
    id: "feature-1",
    icon: buddy,
    title: "AI Mentor",
    content: "Our HighEnd AIMentor  will guide students in their learning journey.",
  },
  {
    id: "feature-2",
    icon: staistics,
    title: "Weakness and strength analysis ",
    content: "Weakness and strength analysis of students and help them in their Learning journey",
  },
  {
    id: "feature-3",
    icon:  robot,
    title: "Face Emotion  Analysis",
    content: "Using AI Models(like OpenCV+DeepFace or AffectivaAPI) to analyze students facial expressions.",
    
  },
];





export const footerLinks = [
  {
    title: "Useful Links",
    links: [
      {
        name: "Content",
        link: "https://www.hoobank.com/content/",
      },
      {
        name: "How it Works",
        link: "https://www.hoobank.com/how-it-works/",
      },
      {
        name: "Create",
        link: "https://www.hoobank.com/create/",
      },
      {
        name: "Explore",
        link: "https://www.hoobank.com/explore/",
      },
      {
        name: "Terms & Services",
        link: "https://www.hoobank.com/terms-and-services/",
      },
    ],
  },
  {
    title: "Community",
    links: [
      {
        name: "Help Center",
        link: "https://www.hoobank.com/help-center/",
      },
      {
        name: "Partners",
        link: "https://www.hoobank.com/partners/",
      },
      {
        name: "Suggestions",
        link: "https://www.hoobank.com/suggestions/",
      },
      {
        name: "Blog",
        link: "https://www.hoobank.com/blog/",
      },
      {
        name: "Newsletters",
        link: "https://www.hoobank.com/newsletters/",
      },
    ],
  },
  {
    title: "Partner",
    links: [
      {
        name: "Our Partner",
        link: "https://www.hoobank.com/our-partner/",
      },
      {
        name: "Become a Partner",
        link: "https://www.hoobank.com/become-a-partner/",
      },
    ],
  },
];

export const socialMedia = [
  {
    id: "social-media-1",
    icon: instagram,
    link: "https://www.instagram.com/",
  },
  {
    id: "social-media-2",
    icon: facebook,
    link: "https://www.facebook.com/",
  },
  {
    id: "social-media-3",
    icon: twitter,
    link: "https://www.twitter.com/",
  },
  {
    id: "social-media-4",
    icon: linkedin,
    link: "https://www.linkedin.com/",
  },
];

export const clients = [
  {
    id: "client-1",
    logo: airbnb,
  },
  {
    id: "client-2",
    logo: binance,
  },
  {
    id: "client-3",
    logo: coinbase,
  },
  {
    id: "client-4",
    logo: dropbox,
  },
];  

export const services = [
  {
    id: 'service1',
    title: 'PDF Summarizer',
    content: 'A PDF Summarizer is a tool that condenses lengthy PDF documents into concise summaries, highlighting key points and essential information',
    icon: pdf,
    link: '/pdf-tools', // Ensure this is the correct path
  },
  {
    id: 'service2',
    title: 'Community',
    content: 'A website which connects all of the students in the community',
    icon: globe,
    link: '/community', // Ensure this is the correct path
  },
  {
    id: 'service3',
    title: 'Pomodoro',
    content: 'A time management method that breaks work into 25-minute intervals (Pomodoros) followed by short breaks.',
    icon: clock,
    link: '/pomodoro', // Ensure this is the correct path
  },
  {
    id: 'service4',
    title: 'FlashCards',
    content: 'Making learning fun, easy, and interesting.',
    icon: flashCards,
    link: '/flashcards', // Ensure this is the correct path
  },
  {
    id: 'service5',
    title: 'ExamBuddy',
    content: 'Guide you during your exams and reduce your stress and help you to score marks.',
    icon: chatbot,
    link: '/chat-interface', // Ensure this is the correct path
  },
  {
    id: 'service6',
    title: 'Face Emotion  Analysis',
    content: 'Using AI Models(like OpenCV+DeepFace or AffectivaAPI) to analyze students facial expressions.',
    icon: robot,
    link: '/chat-interface', // Ensure this is the correct path
  },
];

  // Add more services as needed
// Ensure each service has a unique id and link
