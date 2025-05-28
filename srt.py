import random

def generate_random_srt(filename="random_test.srt", num_subs=10):
    with open(filename, "w", encoding="utf-8") as f:
        for i in range(1, num_subs + 1):
            # Génère un timestamp aléatoire simple (durée 3 secondes par sous-titre)
            start_sec = i * 5
            end_sec = start_sec + 3

            def sec_to_timestamp(sec):
                h = sec // 3600
                m = (sec % 3600) // 60
                s = sec % 60
                return f"{h:02}:{m:02}:{s:02},000"

            start_ts = sec_to_timestamp(start_sec)
            end_ts = sec_to_timestamp(end_sec)

            # Texte aléatoire simple (choix parmi une liste)
            texts = [
                "Hello world!",
                "This is a test.",
                "Random subtitle line.",
                "Testing subtitle generator.",
                "Streamlit is awesome.",
                "Python is great for TTS.",
                "Enjoy coding!",
                "This is line number " + str(i),
                "Have a nice day!",
                "End of subtitles."
            ]
            text = random.choice(texts)

            f.write(f"{i}\n")
            f.write(f"{start_ts} --> {end_ts}\n")
            f.write(f"{text}\n\n")

    print(f"Generated random SRT file: {filename}")

if __name__ == "__main__":
    generate_random_srt()
