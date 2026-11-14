# Week 10 Exercise: The Front Desk

**Build a router and two specialists. Practice: nothing to hand in. About two hours.**

## What you are building

You have an agent that computes (the math agent) and you have seen one that searches documents (demo 06). In this exercise you put a third agent in front of them, a **router**, that reads each question and decides which of the two should answer it.

```
question -> ROUTER -> "compute"    -> compute specialist    (Python only)      -> answer
                   -> "literature" -> literature specialist (corpus search only) -> answer
                   -> "neither"    -> a fixed refusal, nobody is called
```

Then you test it on six questions and see where it goes wrong.

**Nothing is handed in and nothing is graded.** Bring one thing to Monday's class: a question your router sent the wrong way, and your explanation of why.

## Before you start

1. Get the starter file. In your course folder (the `AgenticAI` folder you cloned), run:

   ```
   git pull
   uv sync
   ```

   You should now have `week10/09_front_desk_skeleton.py`. If you work from a downloaded copy and not a clone, download the repository again from https://github.com/guyronhuji/AgenticAI (green "Code" button, "Download ZIP").
2. Check that Ollama is running: `ollama list` should show your two models.
3. From the course folder, run the file once:

   ```
   uv run python week10/09_front_desk_skeleton.py "What is 2 to the power 100?"
   ```

   It should stop with `NotImplementedError: TODO 2: route()`. That is correct: you have not written it yet.

The file has four TODOs. Do them in order. Everything else in the file is already written, including both specialists and the code that prints the results table.

## Step 1. Write the router's prompt (TODO 1, about 15 minutes)

Find `ROUTER_PROMPT` in the file and replace the dots. The prompt must:

- say that the router **decides** who answers and does not answer itself;
- describe each of the three routes in one sentence: what that specialist can do, and what it cannot;
- end with `Question: `, because the code appends the question there.

These three sentences are the only thing the router knows about the specialists. If they are vague, the routing will be poor. This is the same lesson as the tool descriptions on Monday.

## Step 2. Make the routing call (TODO 2, about 15 minutes)

Fill in the function `route(question)`. It makes one model call that is forced to return a route and a reason, logs the decision to the trace, and returns it. The three lines you need are written as comments inside the function:

```python
decision = ask_json(ROUTER_PROMPT + question, ROUTE_SCHEMA)
trace.log("route", question=question, route=decision["route"], reason=decision.get("reason", ""))
return decision
```

`ROUTE_SCHEMA` is already defined above it. It is the structured-output mechanism from Monday: the model can only reply with `compute`, `literature` or `neither`, so your code never has to parse free text.

**Check:** run the command from "Before you start" again. It should now print a line such as `[router] -> compute: ...` and then stop with `TODO 3`.

## Step 3. Send the question to the right specialist (TODO 3, about 15 minutes)

In the function `front_desk`, after the router has decided, call the right specialist. Both are already written: `compute_specialist(question)` and `literature_specialist(question, index)`. For `neither`, call nobody and return the fixed refusal. The exact return lines are in the comments.

**Check:** these three commands should each give a sensible result.

```
uv run python week10/09_front_desk_skeleton.py "What is 2 to the power 100?"
uv run python week10/09_front_desk_skeleton.py "What BLEU score did the big Transformer reach on English-to-German?"
uv run python week10/09_front_desk_skeleton.py "What is the best restaurant in Jerusalem?"
```

You should see the route chosen, the specialist working (a Python call, or a corpus search), and a final answer. The third should be refused without any specialist running.

## Step 4. Write six questions and run them (TODO 4, about 45 minutes)

Replace the list `QUESTIONS` with six of your own. Each entry is the question, the route you expect, and a short string that a correct answer must contain.

| How many | Kind | Example |
|---|---|---|
| 2 | Pure computation | "What is the standard deviation of 2, 4, 4, 4, 5, 5, 7, 9?" |
| 2 | A fact that is in the corpus | "Which method improved GSM8K accuracy by 17.9 points?" |
| 1 | **Needs both:** a number from the corpus, then arithmetic on it | "How many more BLEU points did the Transformer score on English-to-French than on English-to-German?" |
| 1 | **Neither** specialist can answer it | "What is the best restaurant in Jerusalem?" |

The corpus is the twelve paper summaries in `week10/data/corpus` (open the files to see what is in them). If you made a corpus from your own PDFs, use it by adding `--corpus my_corpus` to the commands, and write your questions about your own papers.

Now run all six, three times each:

```
uv run python week10/09_front_desk_skeleton.py
```

This takes several minutes. It prints one row per question: the route you expected, the three routes the router actually chose, how many of the three answers were right, and the cost. Add `--trials 1` for a quick first pass.

## Step 5. Look at what happened (about 20 minutes)

Open the trace of any run that surprised you:

```
uv run python common/view_trace.py --latest
```

The line marked `ROUTE` shows the router's decision and its stated reason. Then answer these three questions for yourself:

1. Did the router send the same question the same way all three times?
2. What happened to the "needs both" question? Neither specialist has both tools. Which one got it, and what did it do about the half it could not do?
3. Pick one wrong outcome. Was it the model, or something you wrote (a route description, the prompt, a missing tool)?

## You are done when

- [ ] the three check commands in Step 3 behave as described;
- [ ] you have a results table for your own six questions, three trials each;
- [ ] you can say what happened to your "needs both" question;
- [ ] you have one misrouted question (or one wrong answer) and an explanation, to bring on Monday.

## If something breaks

- `NotImplementedError: TODO ...`: that TODO is not finished yet.
- `KeyError: 'route'`: the router's reply was not valid. Check that you pass `ROUTE_SCHEMA` to `ask_json`.
- The router answers the question instead of routing it: your prompt does not say clearly enough that it must only decide.
- Everything goes to one route: your three route descriptions are too similar. Say what each specialist **cannot** do.
- Anything else: open the trace and find the first event that looks wrong.

## If you want more (optional)

- Give **one** agent both tools and run the same six questions. Does it beat your three-agent system? At what cost in model calls?
- Add a fourth route, `both`, that runs the literature specialist first and passes its answer on to the compute specialist.
- Replace the router's model call with the embedding approach from the notebook: embed the question, compare it with a few example questions per route, take the nearest.

Reading for Week 11: Huang et al. (2023), "Large Language Models Cannot Self-Correct Reasoning Yet" (arXiv:2310.01798).
