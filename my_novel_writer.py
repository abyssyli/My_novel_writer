from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import datetime

# Load phi-1.5 model and tokenizer
model_name = "microsoft/phi-1_5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

# Load translation model (Chinese → English)
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-zh-en")

# Embedded novel seed (translated English version from user's writing)
novel_seed = """
My life is made of fragments frozen with blood and tears.
From June 22, 2018, to the first snow in Shanghai that winter. From entering UC Santa Cruz in 2019 to fleeing to San Jose during the 2020 wildfires. In fall 2020, I discovered Los Altos and switched to a math major. A brief winter break in Dublin in early 2021 brought me back to Shanghai in March. Then it felt like there were no moments in life worth remembering anymore.
Lately, I often miss the summer of 2020. Maybe it’s the weather and season, but I can never find that version of myself again. So today, let’s revisit that July.
That passionate summer began when I heard that a senior with a green card, who wasn’t as academically strong as I was, transferred to UCLA. I thought I could do it too, so I decided to finish all my remaining GEs that summer. I was still a biochemistry major but fascinated by math, thanks to Professor Mark Eastman. And for some reason, getting good grades in UCSC’s math department was oddly easy. So I kept telling myself: just one more math class, then another, and another...
That summer, I took math23B, which brought me to tears, and met Christian Sabile, who would become a lifelong friend. We went to classes, wrote math, watched the sea and moon from Stevenson field, hunted cougars behind the engineering building, and encountered the harmless foxes. The school emailed us about cougar sightings. Funny, right? If they only told us there were cougars, I’d be scared. But they told us exactly where, so I had to see for myself.
To get to Safeway in town, we crossed fields of cattle. But I hadn’t gone in a while—the dining hall food was great. In the evenings, at Stevenson, we saw herds of deer. The newborns weren’t afraid of humans—one even smiled at me. I don’t know what they’ll go through to become fearful later. I dislike adult deer—self-protective but rude. One saw me and immediately lifted a leg and peed. But more than deer, I hated the turkeys. There was a flying one named Hank, though I never saw it. Most turkeys bit, which I learned the hard way. I nearly got bitten before escaping into the bookstore. One day, a turkey mom crossed the road with her chicks, and while I was distracted, my MacBook Air fell and got dented.
At night, I walked through the redwoods to McHenry Library. Even during break, math PhDs stayed and studied. It was my favorite place on campus. During walks, music was essential. I liked listening to “Stay With Me” by Chanyeol—not because of Goblin, but because I saw an edit from “Bride of the Water God” where Nam Joo Hyuk descended from the sky to catch Shin Se Kyung and said, “Don’t be afraid. You belong to the gods.” God, he was cool. I wished a prince like that could teach me math proofs. Music was also important when running away. One night, I encountered a homeless man at McHenry’s entrance. He had long hair and a beard. I ran immediately, blasting the track “Protect Her” from “Cinderella and Four Knights.” Months later, I realized that guy was my TA in Mathematical Python.
Late at night, the moon came out. I stood on the Stevenson hill, watching the sea and the moon. Along the coastline, town lights twinkled. Compared to the MV of Despacito, where the waves sound romantic, the roar of the real ocean was incomparable. When the moon was fullest, I reached out with my phone to focus—it looked like I held a glowing pearl. I remembered dreaming about the West Coast in a Starbucks on Tibet Road in Shanghai in 2018. Lifting my head, I realized everything had become the past. I stood quietly in the breeze, forgetting time again and again.
Time took not only that secret land between the sea and forest—my most unforgettable haven in America—but also someone I met in the summer of 2020: Kaden.
She was my writing TA. Well, they were. I went to their sessions just to get an A and keep my 4.0. But it turned out I was always the only one who showed up. We got close. She was fully white. I always believed there was an unbridgeable cultural gap between us. I also knew my writing was terrible. But Kaden kept encouraging me.
At first, I thought it was politeness, a top-class white person’s gentleness. Later I realized she cared especially about me. We added each other on Facebook. That’s when I found out she was transgender. Before transitioning, she had a sweet girl’s name. I felt a bit of pain for her—but if this was what she wanted, of course I’d support her.
We became close. My writing improved. While waiting to see her, I often listened to Miyuki Nakajima’s “With.” Later, I found Sammi Cheng’s version—“Romeo and Juliet of Sarajevo.” It claimed love transcends race and religion. I rolled my eyes. A song’s just a song. But then, I found out she liked me.
Her compliments shifted—from calling me a diligent student to saying I was a hot girl from Asian. Then, wildfires reached our campus. She wanted to bring me to her home that night. That’s when I knew something had changed.
I was scared. Even though I often thought of her while writing essays and memorizing words.
Even without thinking of my mom, I knew her mom would go mad knowing her white daughter was with an Asian girl. I couldn’t make Kaden’s mom upset. So we couldn’t become anything more.
In August 2020, when the fire reached Santa Cruz, I stayed in a seaside tent with other international students. I didn’t respond to her invitation. Then I escaped to San Jose.
I never saw Kaden again. Everyone went home. Then COVID swept through campus. No one lived on campus anymore.
I didn’t even hug her.
Sometimes, I wonder—if I had gone to Maryland with her, would I be someone else now?
But there are no what-ifs. So, that summer remains frozen. The wind and melodies carry a quiet sadness. And no one ever liked me again the way Kaden did in the U.S..
"""

# Output file
output_file = "abyssyli_generated_novel.txt"

# Generation function
def generate_abyssyli_style(topic: str, max_tokens=300):
    if any(u'\u4e00' <= ch <= u'\u9fff' for ch in topic):
        topic = translator(topic, max_length=512)[0]['translation_text']

    prompt = f"""
You are a writer named abyssyli. Your style is poetic, nostalgic, emotionally restrained.
You write in fragmentary memories, through nature, time, silence, and personal loss.

Here is your previous work:
{novel_seed}

Now, write a new paragraph in the same style.
Topic: "{topic}"
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.9,
            top_p=0.95
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated = result.replace(prompt, "").strip()

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(f"\n---\n🕓 {timestamp}\n🎯 Topic: {topic}\n\n{generated}\n")

    print("\n📝 Generated:\n")
    print(generated)
    print(f"\n✅ Saved to: {output_file}")

# CLI entry point
if __name__ == "__main__":
    user_topic = input("请输入你想生成小说段落的主题（可用中文或英文）：\n> ")
    generate_abyssyli_style(user_topic)