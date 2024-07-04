#!pip install openai==0.28
import openai
import pandas as pd
class MoviePlotGenerator:
    def __init__(self, api_key):
        self.api_key = api_key
        openai.api_key = self.api_key

    def get_completion(self, prompt, model="gpt-4"):
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=1,  # degree of randomness of the model's output
        )
        return response.choices[0].message["content"]

    def create_prompt(self, plot, set_actseq):
        return f"""
        Create a movie plot with {plot} based on: {set_actseq}
        """

    def main(self):
        print("Enter the plot description:")
        plot = input()
        print("Enter the set of activity sequences:")
        set_actseq = input()
#example
  # "Spy thriller, grieving, coping, intimidate, fail, assault, kill, steal, recognizing, refuse, inform, beats, berates, fear, recover, refuse, kill, put bounty, lodge, warn, infiltrate, confront, retreat, rest, sneak, subdue, reveal, escape, destroy, assess, assault, threaten, intervene, encourage, witness, torture, taunt, execute, inform, race, fight, wound, resign, watch, break, treat, adopt, walk, unsure, approach, ready. Events: finishing mission, sent to Prague, stop rogue agent, steal NOC list, mission fails, list stolen, team killed, escape, return to safe house, realize meaning of Job 314, meet Max, warn about fake NOC list, escape raid, obtain real NOC list, recruit Luther Stickell and Franz Krieger, infiltrate CIA headquarters, steal authentic list, escape to London, arrest of Hunt's mother and uncle, contact Kittridge, trace the call, Phelps resurfaces, exchange with Max, reveal Phelps as mole, confrontation with Phelps, helicopter chase, explosion, Kittridge takes Max into custody, recover the NOC list, reinstated in IMF, unsure about returning, approached for a new mission.",
       
  # "action, superhero, robs, betraying, killing, reveals, escapes, ally, eliminate, supports, retire, pursue, interrupts, offers, kill, concealed, fleeing, find, returns, apprehend, accept, threatens, attacks, continues, reveals, targets, throws, rescues, struggles, understand, relish, claims, lure, deduced, reveals, separates, kills, severely burned, escapes, extracts, burns, deduces, expose, blow up, evacuate, struggles, meets, persuading, defers, killing, grips, reveals, rigged to explode, refuse to kill, subdues, refuses to kill, arrest, incorruptible, corrupts, takes hostage, blames, falls to death, takes blame, persuades, conceals, burns, mourns, launch manhunt, fear, grief, intimidation, assault, loss, hatred, revenge, captivity, danger, warning, subduing, taunting, hope, sacrifice, panic, disbelief, struggle, guilt, negligence, death, destruction, trust, concealment, mourning.",

        prompt = self.create_prompt(plot, set_actseq)
        response = self.get_completion(prompt)
        print(response)

# Replace 'api_key' with actual OpenAI API key
api_key = '  '
movie_plot_generator = MoviePlotGenerator(api_key)
movie_plot_generator.main()
