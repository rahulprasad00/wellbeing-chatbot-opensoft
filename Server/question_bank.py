import json

question_bank = {
  "Average_Vibe_Score": {
    "description": "Explores how employees perceive their overall emotional well-being and satisfaction at work.",
    "questions": [
      "How do you think your overall mood compares to your general sense of well-being at work?",
      "How do you feel about the trend in your mood at work—has it been improving or declining?",
      "Does your level of mood align with your overall job satisfaction? Why or why not?",
      "What aspects of your job contribute most to your mood, either positively or negatively?",   
      "Do you think your mood at work affects how engaged or productive you feel?"
    ]
  },
  "Latest_Vibe_Score": {
    "description": "Focuses on current mood and short-term factors influencing emotional state at work.",
    "questions": [
      "Does your current mood at work match how you generally feel during the day?",
      "What are some factors you believe have contributed to your mood recently?",
      "Have you noticed any patterns in your mood based on workload or deadlines?",
      "Do you feel more supported by your colleagues when your mood is better?",
      "Would you say your mood accurately represents your day to day experience at work?"
    ]
  },
  "Total_Rewards_Received": {
    "description": "Evaluates employee satisfaction with recognition and rewards for contributions.",
    "questions": [
      "Do you feel adequately rewarded for your contributions, considering your Total Rewards Received?",
      "How do you perceive the balance between the rewards you have received and the effort you’ve put in?",
      "What type of rewards or recognition do you find most meaningful?",
      "Do you feel that the reward system is fair and transparent across the organization?",
      "Have you ever felt that your contributions deserved more recognition than you received?"
    ]
  },
  "Total_Reward_Points": {
    "description": "Examines the relationship between reward points and employee motivation or value perception.",
    "questions": [
      "Have you noticed any correlation between your Total Reward Points and your overall motivation?",
      "Do you think your reward points accurately capture your hard work and dedication? Why or why not?",
      "What role do reward points play in shaping your long‑term motivation at work?",
      "Do you believe the reward system considers qualitative contributions like teamwork and leadership, or is it only based on quantitative performance?",
      "Would receiving more frequent, non‑monetary rewards make you feel more appreciated?"
    ]
  },
  "Average_Reward_Points": {
    "description": "Assesses how effectively average reward point distribution supports recognition and morale.",
    "questions": [
      "Do you think increasing the number of reward points per achievement would improve engagement?",
      "Do you feel the average reward points given are aligned with the value of the work recognized?",
      "Have you ever felt that a task or achievement deserved more reward points than you received?",
      "How does the average reward points distribution affect your motivation to contribute beyond your core responsibilities?",
      "Would more consistent recognition through reward points improve your sense of accomplishment?"
    ]
  },
  "Days_since_last_reward": {
    "description": "Looks at how the timing of recognition affects employee engagement and morale.",
    "questions": [
      "How has the time since your last reward affected your feelings of engagement at work?",
      "How often do you think an ideal reward cycle should be to keep employees engaged?",
      "Does the timing of your last reward impact how valued you feel in your role?",
      "Have you ever felt demotivated due to the lack of rewards or recognition over time?"
    ]
  },
  "Sick_Leaves": {
    "description": "Explores employee comfort and organizational support around health-related absences.",
    "questions": [
      "Do you believe your current leave allowance supports a healthy work-life balance?",
      "Have you ever hesitated to take a leave due to workload or team pressure?",
      "Do you feel that taking regular breaks helps you maintain long‑term motivation at work?",
      "Do you feel comfortable taking leave when you need it, or do you feel guilty about it?"
    ]
  },
  "Annual_Leaves": {
    "description": "Assesses perceptions of annual leave policy and its effectiveness in supporting well-being.",
    "questions": [
      "How do you feel about the frequency and availability of your leave options?",
      "Do you feel that the current leave policy supports your personal and professional needs adequately?",
      "How do you usually decide when to take a leave, based on workload, personal needs, or other factors?",
      "Does the process of requesting and approving leave feel smooth and fair to you?",
      "Have you ever felt the need for additional leave types that aren’t currently available?"
    ]
  },
  "Casual_Leaves": {
    "description": "Focuses on usage patterns, flexibility, and impact of casual leave on work-life balance.",
    "questions": [
      "Is there a specific type of leave that you tend to use more frequently, and why?",
      "Would you take more leave if you felt it wouldn't impact your career progression?",
      "How does taking leave affect your workload when you return? Do you feel overwhelmed catching up?",
      "Do you believe your company’s leave policy is competitive compared to industry standards?",
      "If you could improve one aspect of the leave policy, what would it be?"
    ]
  },
  "Unpaid_Leaves": {
    "description": "Evaluates how financial and policy factors influence the use of unpaid leave.",
    "questions": [
      "Would you take more unpaid leaves if financial constraints weren’t a factor?",
      "Have you ever had to work while on leave due to urgent tasks? If so, how did it impact you?",
      "Do you feel comfortable requesting unpaid leave when needed?",
      "How does the option (or lack) of unpaid leave affect your ability to manage personal situations?",
      "Would clearer guidelines or policies around unpaid leaves encourage you to use them more when necessary?"
    ]
  },
  "Days_since_last_leave": {
    "description": "Assesses how time between leaves impacts energy, well-being, and productivity.",
    "questions": [
      "Has the number of leaves you’ve taken recently impacted your overall well‑being?",
      "Do you feel that the current leave system supports proactive leave-taking rather than reactive leave-taking?",
      "Do you feel refreshed and recharged after your most recent leave?",
      "How does the time since your last leave affect your current energy and motivation levels?",
      "Have you ever postponed taking leave even when you felt it was needed? What held you back?"
    ]
  },
  "Average_Work_Hours": {
    "description": "Explores the impact of workload, autonomy, and time management on well-being and performance.",
    "questions": [
      "Do you feel that your workload is evenly distributed throughout the week, or do you experience periods of extreme pressure?",
      "Do you feel you have enough time to focus on deep work, or are you frequently interrupted by meetings and messages?",
      "How do you feel about the level of autonomy you have in managing your workload?",
      "How does your current workload affect your ability to disconnect from work outside office hours?",
      "Do you believe your workload allows for a healthy work‑life balance, or does it need improvement?"
    ]
  },
  "Onboarding_Feedback": {
    "description": "Evaluates new hire onboarding experiences in terms of support, clarity, and integration.",
    "questions": [
      "How well did your onboarding experience prepare you for your current role?",
      "Did you feel adequately supported during your initial training and onboarding period?",
      "Were there any gaps in your onboarding experience that you feel could have been improved?",
      "How would you rate your overall onboarding experience in terms of making you feel welcomed and integrated into the team?",
      "What aspects of the onboarding process would you recommend improving for future new hires?"
    ]
  },
  "Mentor_Assigned": {
    "description": "Assesses the effectiveness and impact of mentorship during the initial employment period.",
    "questions": [
      "Do you feel your mentor played a significant role in your adjustment to the company?",
      "What additional guidance or mentorship would help you achieve your career goals?",
      "How often do you interact with your assigned mentor, and is that frequency effective?",
      "Do you feel comfortable reaching out to your mentor with questions or concerns?",
      "Has your mentor helped you understand the company culture and unwritten norms?"
    ]
  },
    "Initial_Training_Completed": {
      "description": "Insights into how well initial training prepared employees for their roles.",
      "questions": [
        "Do you feel your initial training prepared you for the challenges you face in your role?",
        "Were there any gaps in your training that you had to fill on your own?",
        "How relevant was the initial training content to your day-to-day responsibilities?",
        "Would additional or follow-up training sessions improve your confidence at work?",
        "Do you feel the training program encouraged collaboration or connection with other new hires?"
      ]
    },
    "Average_Manager_Feedback_Score": {
      "description": "Understanding the quality and impact of managerial feedback on employee performance.",
      "questions": [
        "What feedback from your manager has been most helpful in improving your performance?",
        "Do you feel you have enough opportunities to demonstrate your performance and receive constructive feedback?",
        "How do your work habits and feedback from your manager influence your career growth and goals?",
        "Do you receive timely and constructive feedback that helps you improve your work?",
        "Do you feel confident that your manager understands your strengths and areas for growth?"
      ]
    },
    "Average_Performance_Rating": {
      "description": "Employee perception of performance evaluations and their fairness.",
      "questions": [
        "Do you feel your Average and Latest Performance Ratings accurately reflect your work contributions?",
        "Do you believe the performance evaluation system is transparent and fair across the company?",
        "What specific actions are you taking to improve your performance in areas that need development?",
        "How well do you think your performance evaluations align with your actual contributions and efforts?",
        "How satisfied are you with the frequency and depth of performance reviews at your company?"
      ]
    },
    "Promotion_Consideration_Ratio": {
      "description": "Explores employee views on promotion readiness and growth opportunities.",
      "questions": [
        "How do you feel about your potential for promotion, given the current Promotion Consideration Ratio?",
        "Do you feel motivated to achieve a promotion, and are you receiving the right support to reach that goal?",
        "How clearly do you understand the criteria used for promotions in your organization?",
        "Do you feel there is a clear path for career advancement in your current role?",
        "How frequently do you discuss career growth and future opportunities with your manager?"
      ]
    },
    "Days_since_last_activity": {
      "description": "Gauges recent employee engagement and activity levels.",
      "questions": [
        "When was the last time you felt actively engaged in work-related activities?",
        "Do you feel that recent lulls in activity have impacted your motivation?",
        "Have you noticed any patterns in your engagement levels over the past few weeks?",
        "What factors have contributed to periods of lower activity recently?",
        "Do you feel supported in staying engaged with your team and responsibilities?"
      ]
    },
    "Latest_Work_Hours": {
      "description": "Assesses workload, work patterns, and their effect on well-being.",
      "questions": [
        "Do your most recent work hours reflect a healthy work-life balance?",
        "Have your latest working hours felt more intense or more relaxed than usual?",
        "How do your recent hours impact your energy and focus during the workday?",
        "Do you feel that your latest workload was manageable within your working hours?",
        "Does your recent work schedule give you enough time for rest and recovery?"
      ]
    },
    "Average_Team_Messages_Sent": {
      "description": "Measures team communication through messaging and collaboration habits.",
      "questions": [
        "Do you feel that your average messaging activity reflects your level of team collaboration?",
        "How often do you engage with your team through chat or messaging tools?",
        "Do you feel comfortable initiating conversations or contributing in team messages?",
        "How do team chats help you stay aligned with daily goals or updates?",
        "Do you feel encouraged to share ideas or feedback through team messages?"
      ]
    },
    "Days_since_joining": {
      "description": "Explores employee integration, growth, and well-being over time.",
      "questions": [
        "How would you describe your journey since joining the organization?",
        "Do you feel well-integrated into your role considering the time you've been here?",
        "Have your responsibilities evolved since your first few weeks at work?",
        "Do you feel more confident and capable now than when you first started?",
        "How has your relationship with your team changed since joining?"
      ]
    },
    "Average_Emails_Sent": {
      "description": "Evaluates email communication patterns and efficiency at work.",
      "questions": [
        "Do you feel that your email activity reflects your overall communication style at work?",
        "How often do you use emails to communicate progress or escalate issues?",
        "Have you found email to be an efficient way to manage work tasks?",
        "Do you feel that email responses are timely and helpful from others?",
        "How do you balance emails with other communication tools like chat or meetings?"
      ]
    },
    "Latest_Performance_Rating": {
      "description": "Focuses on recent performance reviews and their impact on motivation and development.",
      "questions": [
        "Do you feel your most recent performance rating reflects your actual contributions?",
        "Were you satisfied with the way your latest performance review was communicated?",
        "Did the feedback associated with your last performance rating feel actionable and fair?",
        "Did your latest rating align with your personal expectations or goals?",
        "How did your most recent performance rating affect your motivation?"
      ]
    },
    "Latest_Promotion_Consideration": {
      "description": "Examines fairness, communication, and impact of the most recent promotion discussions.",
      "questions": [
        "Do you feel your latest promotion consideration was handled fairly and transparently?",
        "Were you given feedback related to your recent promotion opportunity?",
        "How did your latest promotion consideration align with your personal goals?",
        "Do you feel you received enough support in preparing for your next role?",
        "Was the reasoning behind your most recent promotion decision clearly explained?"
      ]
    }
}

