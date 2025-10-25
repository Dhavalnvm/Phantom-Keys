# PHANTOM KEYS
## The Virtual Piano That Judges Your Every Keystroke

---

### Oh, You Want to Play Piano? How Adorable.

Welcome to **PHANTOM KEYS** - the only virtual piano with enough attitude to tell you what you REALLY sound like. No sugar-coating. No participation trophies. Just you, your webcam, and the cold, hard truth about your "musical abilities."

---

## What Is This Masterpiece?

PHANTOM KEYS is a gesture-controlled virtual piano that tracks your hand movements via webcam. It's like Guitar Hero, except:
- No guitar
- No hero
- Just you waving at a camera like you're hailing a taxi in the rain

### Features (That You'll Probably Waste):
- **5 Finger Tracking** - Yes, even your pinky (we see you neglecting it)
- **Falling Notes** - Like your GPA, but visual
- **Particle Effects** - To distract from your lack of rhythm  
- **Score System** - So you can see numerically how disappointing you are
- **2.5 Octaves** - More keys than you'll ever successfully hit
- **Sarcastic Commentary** - Because you need someone to keep it real

---

## What You'll Need

### Minimum Requirements:
- **A functioning webcam** (that your cat hasn't broken)
- **Python 3.8+** (because apparently we're doing this)
- **10 fingers** (or fewer, we don't judge... much)
- **Realistic expectations** (LOL nevermind, too late)

### Recommended Requirements:
- **Actually knowing what a piano is**
- **Basic hand-eye coordination** (we can dream)
- **The ability to count to 8** (for the keys)
- **Thick skin** (seriously, this piano is MEAN)

---

## Installation

### Step 1: Accept Your Fate
```bash
pip install opencv-python mediapipe pygame numpy torch
```

### Step 2: Download the Code
```bash
# Clone it, download it, steal it from your friend, whatever
# We both know you're going to give up in 10 minutes anyway
```

### Step 3: Actually Run It
```bash
python phantom_keys.py
```

**IF** it doesn't work (and let's be honest, it won't on the first try):
```bash
# Run the setup script like a normal person
python setup.py
```

---

## How to "Play" (And I Use That Term Loosely)

### Basic Controls:
- **Wave your hands** - Like you're conducting an orchestra (that doesn't exist)
- **Move finger DOWN** - To *attempt* to play a note
- **Press 'S'** - Switch between Song Mode and Freeplay (when you inevitably fail)
- **Press 'Q'** - Quit (admitting defeat already?)
- **Press 'R'** - Reset (for your third attempt)
- **Press 'G'** - GPU glow effects (because if you're failing, at least look cool)

### Song Mode:
Notes fall from the sky like your dreams. Try to hit them. You won't. But try.

### Freeplay Mode:
Play whatever you want! Wrong notes included. Free of charge.

---

## The Keys (That You'll Miss)

```
White Keys:  C  D  E  F  G  A  B  C
           ┌──┬──┬──┬──┬──┬──┬──┬──┐
Black Keys: │C#│D#│  │F#│G#│A#│  │  │
           └──┴──┴──┴──┴──┴──┴──┴──┘
```

See those black keys? Yeah, don't worry about them. You're not ready.

---

## Finger Color Coding

Because apparently you need a color-coded map to know where your own fingers are:

- **Thumb** - Orange (the one you use for scrolling TikTok)
- **Index** - Green (your "pointing at things" finger)
- **Middle** - Yellow (we know what you use this one for)
- **Ring** - Magenta (the one with the ring you can't afford)
- **Pinky** - Pink (completely useless, just like in real life)

---

## Scoring System

### How You're Judged:
- **Hit a note**: +100 points *(wow, basic functionality)*
- **Miss a note**: Combo reset *(saw that coming)*
- **Combo**: Consecutive hits *(good luck with that)*
- **Best Combo**: Your peak before the inevitable collapse

### Score Interpretation:
- **0-500**: Did you even try?
- **500-1000**: Participation award energy
- **1000-2000**: Okay, maybe you've touched a piano before
- **2000+**: Are you... actually good? (Doubt it, but respect)

---

## Troubleshooting

### "The camera isn't working!"
> Have you tried... turning it on? Revolutionary, I know.

### "It's not detecting my hands!"
> Perhaps try having... hands? Also, improve your lighting. You're not a vampire.

### "The notes are falling too fast!"
> Sounds like a YOU problem. But fine, edit the code. Line 145. You're welcome.

### "I can't hit any notes!"
> We know. The piano knows. Everyone knows.

### "The score isn't going up!"
> That's because you're not hitting the notes. See previous answer.

### "This README is too sarcastic!"
> Finally, something we agree on. Now go practice.

---

## Pro Tips (From People Better Than You)

1. **Use good lighting** - You're not filming a Christopher Nolan movie
2. **Extend your fingers** - Like you're reaching for success (that you won't achieve)
3. **Move decisively** - Hesitation is for people who read instructions
4. **Start with Freeplay** - Baby steps for baby skills
5. **Don't give up** - (You will, but we're legally required to say this)

---

## Achievements (You Won't Unlock)

- **Baby Steps** - Hit 1 note (Congrats on the bare minimum)
- **Mediocre** - Score 500 points (Still not impressive)
- **Pianist** - Score 1000 points (Okay, you practiced)
- **On Fire** - 10x combo (Wait, is that legal?)
- **Phantom Maestro** - Score 3000+ (Screenshot or it didn't happen)
- **Rage Quit** - Close the app within 2 minutes (Most common achievement)

---

## Suggested First Song

**"Mary Had a Little Lamb"**

Because if you can't play a song designed for literal children, we have bigger problems.

```
E D C D E E E
(It's literally the first 7 notes. You got this. Maybe.)
```

---

## FAQ

### Q: Is this piano mean to me?
**A:** Yes. Next question.

### Q: Will this make me better at piano?
**A:** No. But it'll make you better at accepting criticism.

### Q: Why is it called Phantom Keys?
**A:** Because your ability to play is... *ghostly*. Non-existent. Phantom.

### Q: Can I turn off the sarcasm?
**A:** Can you turn off your inability to keep rhythm? Exactly.

### Q: Does this work with a real keyboard?
**A:** No. Go buy a real piano if you're so talented.

### Q: I'm offended by this README.
**A:** Congratulations on having feelings. Now go practice.

---

## What To Expect

1. You'll open the app
2. You'll wave at the camera
3. You'll miss every note
4. You'll blame the program
5. You'll try again
6. You'll miss again
7. You'll question your life choices
8. You'll close the app
9. **Tomorrow you'll be back**

(We both know step 9 is inevitable)

---

## Technical Specifications (For The Nerds)

### What's Actually Happening:
- **MediaPipe Hands** tracks your pathetic finger movements
- **OpenCV** captures video of your failures in HD
- **PyGame** generates sounds you'll never do justice
- **NumPy** calculates exactly how off-tempo you are
- **PyTorch CUDA** (optional) - Makes it faster to witness your mistakes

### Performance Metrics:
- **Latency**: Sub-100ms (faster than your reaction time)
- **Frame Rate**: 60 FPS (60 opportunities per second to fail)
- **Accuracy Tolerance**: ±50 pixels (generous, considering your coordination)
- **Supported Hands**: 2 (though you can barely control 1)

### File Structure:
```
phantom_keys/
├── The_Awakening.py        # The version which can run on cpu
├── The_roasting.py         # The one which has the actual game and works with gpu
├── Phantom_mode.py          # Just a file with controls the time duration of the note with hand gestures(ps : The keyboard is your notes)
├── setup.py                 # Your first failure point
├── requirements.txt         # Things you'll install wrong
└── README.md               # This masterpiece
```

---

## Advanced Features (You'll Never Use)

### GPU Acceleration:
If you have an NVIDIA GPU, the program will run faster. This means you'll fail faster. Efficiency!

### Velocity Sensitivity:
The harder you move your finger, the louder the note. Great for expressing your frustration.

### Multi-Hand Support:
Track both hands simultaneously. Twice the hands, twice the mistakes!

### Particle Effects:
Pretty explosions when you hit a note. Finally, something you're good at - making things explode.

---

## System Requirements Reality Check

### What The Docs Say:
- Python 3.8+
- 4GB RAM
- Webcam
- Basic computer

### What You Actually Need:
- Patience (you don't have this)
- Rhythm (definitely don't have this)
- Coordination (LOL)
- Willingness to learn (we'll see)
- Google (to figure out what went wrong)

---

## Known Issues

### Issue #1: User Error
- **Symptom**: Nothing works
- **Cause**: You
- **Fix**: Read the instructions

### Issue #2: Unrealistic Expectations
- **Symptom**: "Why am I not good immediately?"
- **Cause**: Life doesn't work that way
- **Fix**: Practice (novel concept, we know)

### Issue #3: Blame Displacement
- **Symptom**: "This program is broken!"
- **Cause**: No, you're just bad
- **Fix**: Accept it and move on

---

## Version History

### v1.0 - "The Awakening"
- Initial release
- Added basic functionality
- Realized users would struggle
- Added extra sarcasm to compensate

### v1.1 - "The Roasting"
- Increased sarcasm by 300%
- Added more insulting messages
- Users still can't play
- As expected

### v2.0 - "Phantom Mode"
- Full sarcasm deployment
- Zero tolerance for excuses
- Maximum judgment
- Still can't fix your rhythm

---

## Contributing

Want to contribute? Sure! We accept:
- Bug fixes (not that you'll find any)
- Performance improvements (won't help your performance)
- More sarcastic messages (now we're talking)
- Excuses (just kidding, we don't accept those)

### Contribution Guidelines:
1. Code must work (unlike your playing)
2. Comments must be helpful (unlike your technique)
3. No removing the sarcasm (that's literally the point)
4. Test your changes (something you should do with your life choices)

---

## Credits

**Created by:** Someone with too much time and not enough patience for your excuses

**Inspired by:** Every piano teacher who ever said "just practice more"

**Dedicated to:** Everyone who said "how hard can it be?" (Very. The answer is very.)

**Special Thanks:** 
- MediaPipe (for the hand tracking you'll abuse)
- OpenCV (for the camera that sees everything)
- PyGame (for the sounds you'll butcher)
- PyTorch (for the GPU you probably don't have)
- Your ego (RIP, it died on line 3)

---

## Legal Disclaimer

This software is provided "AS IS" without warranty of any kind.

We are not responsible for:
- Hurt feelings
- Bruised egos
- Existential crises
- Sudden realizations about your musical abilities
- Keyboard rage
- Webcam violence
- Relationships damaged by your practice sessions
- Your neighbors' complaints
- The truth

Use at your own risk. Both the software and your dignity.

---

## Support

Need help? Of course you do.

**Option 1:** Read the documentation  
**Option 2:** Google it  
**Option 3:** Try again  
**Option 4:** Accept defeat  

There is no Option 5.

---

## Final Thoughts

Look, we believe in you. 

Just kidding. We don't. 

But play anyway. Prove us wrong. We DARE you.

Remember: Every expert was once a beginner who refused to give up.

You'll probably give up. But hey, prove us wrong.

---

## Motivational Quote

> "The piano ain't got no wrong notes."
> - Thelonious Monk

(Clearly he never heard you play)

---

## Actual Useful Information (Hidden At The End)

Okay, real talk for a second:

This is actually a fun project to learn computer vision, gesture recognition, and real-time audio processing. The sarcasm is just to keep you entertained while you practice. 

The more you use it, the better you'll get. The hand tracking is legitimately good, the audio synthesis is decent, and the visual effects are pretty cool.

So yeah, we joke, but... actually try it. It's kinda neat.

(But you'll still miss notes. That part wasn't a joke.)

---

## Contact

Questions? Comments? Complaints?

Send them to: /dev/null

(We're kidding. Kind of. Open an issue on GitHub if you actually need help.)

---

**PHANTOM KEYS**  
*"The only piano that plays back... at you."*

Now stop reading and start playing.

Or don't. The piano doesn't care about your feelings.

But your future self does.

---

### P.S.
Yes, this README is unnecessarily mean.  
Yes, you're still going to use it.  
Yes, we both know why.  

Now stop reading and start playing.

### P.P.S.
Seriously, go practice.

### P.P.P.S.
We're watching. (The webcam, remember?)
