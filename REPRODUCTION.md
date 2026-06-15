# Future Work and Replication Document

This document is for developers who might work on or contribute to this project in the future. First I will expain how to run the program end to end then hand you context, background, and ideas about the project. things like where we started, what we explored, and where I think it's going, along with the tips and hard-won lessons I wish someone had handed me. You can read it straight through like a notebook to get my whole perspective, or you can jump to the section you need and use it as a reference. Either way, if you have questions I'd rather talk than have you guess, so a quick Zoom call will probably save hours.

One note on how this is laid out. The very next section is the practical part: how to actually get the thing running on your machine. If you're here to replicate results and that's all you need, you can stop after it. Everything past that is the deeper story, the why and the how and if you are attempting to contibute more.

# Getting it running 

We have 4 steps that you will need to run these are all contained in 3 diffrent programs. This programs idea is to be able to predict devlopers disengament prehemptivly and give insights on how to help effects of the departur. This is why we need 4 steps. Step one is getting the past data on devlopers to predict their future. Then we need to take our raw data and give some structre to it so our computers can learn. then we set the computers to model and train on the data we collced then trained. Lastly we now have all the data on activty, the fucture forcast of actity and insights from the strcuted data we made and need to serve it to the user.

## Step 0 - Enrivoment Setup

This project has two environments: **Python** (for data collection) and **Node.js** (for the dashboard). Python should be 3.9+. If you have an enviroemnt you want to use you can skip this step.

### 0a. Create the environment

```bash
   pip install -r requirements.txt
```

### 0b. OR Anaconda Python environment

```bash
   conda env create -f environment.yml
   conda activate distrac-env
```
  
### 0b. Node.js environment (for the dashboard)

1. Install Node.js: https://nodejs.org/ (v16+ recommended)
2. Install dependencies:
```bash
   npm install
```

3. Verify it worked:
```bash
   npm --version
```


## Stage 1 - Collect raw GitHub data

Our first step is to pull all the data we need for each project from the GitHub API and writes them to disk @ `Organizations/<org>/<repo>/`. To do this we need 2 files.

### 1a. API tokens File

Create the tokens file to find this go to GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic) (one classic personal access token per line). Each token needs the `repo` and `read:user` scopes. The collector rotates across tokens: one token runs serially; two or more run in parallel, one worker per token. confirm the tokens-file path in `Settings.py` before your first run.

### 1b. Repositories File

The repos file holds one `org/repo` per line. Lines starting with `#` are ignored, and full `https://github.com/org/repo` URLs are accepted too:
    Rdatatable/data.table
    JabRef/jabref
    facebook/react 

### 1c. Run the extractor

Last step!

```
python "Data Collection/CommitExtractorV3.py"
```
For each repo we collects three streams issues, PRs, and commits and then does a commit org-wide sweep that writes `_org_activity.csv` (needed later for the cross-repo predictors). You'll see a progress bar one stream at a time. Large repo (thousands of contributors) can take hours on issue/PR collection 57 Repos takes about a week so let them run overnight with many keys.

### 1d. Extra features: Stopping, resuming, and resetting

- **Resume:** just re-run the same command. Each repo stores durable progress in  `data_cursor.json` / `scheduler_state.json`, so a re-run skips finished streams and continues mid-stream from the last page. Safe to Ctrl-C and restart anytime.
- **A repo errors out:** the run logs `[ERROR] ... Skipping <repo>` and continues to the next repo. Check the tail of the log for any skipped repos and re-run to retry them.
- **Force a clean re-pull of a stream** (e.g. a corrupted/partial CSV):

      python "Data Collection/CommitExtractorV3.py" --reset-stream issues_with_timeline
      python "Data Collection/CommitExtractorV3.py" --reset-all   # all three streams

  Valid stream names: `issues_with_timeline`, `prs_with_comments`, `commits`. Reset applies to **every repo in the repos file**. To reset just one repo, trim the   repos file to that single line first. When this stage finishes, each project folder contains: `commit_list.csv`, `per_file_commits.csv`, `issues.csv`, `issue_activity.csv`, `prs_repo.csv`, `prs_comments.csv`, `repo_tree.csv` (plus the cursor/scheduler JSONs).


## Step 2  Process the data into a model-ready table

For the second step after we have the data need to process and transform it from raw CSV's from our last step into enriched tables per repo. Find the final product here 'Organizations/<org>/<repo>/Results/all_users_labeled_timeline.csv' after you run it. 

### Run
1. Launch the app: `streamlit run Dashboard/DemoAppV2.2.py`
2. Fill the queue. Simplest path is the **Add Training Repos** and **Add Test Repos**
  buttons (or **Add all to Queue**), which pull from `repo_split.csv`
3. Press **Predictors** (recommended). This walks every queued repo through the whole
  pipeline — truck factor → daily timeline → state labeling → social-technical,
  project-health, and knowledge-distribution features → merged enriched timeline.
4. The **Responce** button (spelled that way in the UI) is the lighter version: it produces
  the `state` response with only daily activity counts as predictors. Use **Predictors**
  for the full feature set the model actually trains on.



Tips: Caching is on by default a repo whose enriched timeline already exists is reloaded, not recomputed. To force a fresh run, switch on **Enable per-step overwrite controls** and toggle the specific step. The Social-Technical Network step is the slow one. Process BOTH train and test repos  becuase Step 3 reads every processed repo from `repo_split.csv`, so any repo you skip now will be missing at training time.


## Step 3 - Train the model and generate predictions

Now that we have all the transformed data we can start training a model. We only need a few things for this step you can find some setting on the website, you can change the code for Cross Validation and the train/test split lives in `repo_split.csv` (one `org/repo` per row with its train/test label). Both this step and the DisTrac read it via `cfg.load_repo_split(...)`. The model and DisTrac know what repos to process from this "repo_split.csv" file it loads straight from disk NOT from the Step 2 queue.

### Run
1. In the **Step 6: Predict Inactivity** section, set the **Forecast horizon**
  (7 / 14 / 21 / 30 days ahead) 
2. Switch on **Overwrite cached model** if you are attempting to retrain from scratch instead of loading the saved model.
3. Set the **Prediction framing** (Binary state is the recommended default): 4-class means the model has 4 final nodes. each node represents the predictor of each class in 14 day horison(Active, Non Coding, Inactive and gone). If you want to represent the models output use a diffrent graph that shows the probablit of each class. The binary state compresses the 4 states into a single "active or inactve" insted of 4 diffrent classes. 
4. Press **Run Inactivity Prediction**. The app loads the train and test labeled timelines from disk per the split, builds the response column for your chosen horizon, and prepares the feature matrix.

IMPORTANT press **Proceed to Training** after loading and checking data. With no cached model present, it runs a hyperparameter search on the train set, then runs inference on each test repo and writes predictions to `Organizations/<org>/<repo>/<model folder>/test_df.csv` (plus the parquet the DisTrac app reads). 
  
Tips: On load you'll see "Training on N developers from M train repos" and the same for the test set confirm those counts match what you processed in Step 2. Mine were around ~207 devlopers for traning set and ~20 test set. The **Audit G — Train vs Test column cross-check** lists every predictor and whether it's present in both sets. Any column in train but missing from test is flagged  "← WILL CRASH". The class-balance line shows the response distribution.

### Possible errors
- **"No training data found"** stops the step on purpose: none of the train repos have a  processed timeline on disk. Go back to Step 2, run **Predictors** for the train repos, and retry. A column flagged in Audit G means a test repo was processed with an older feature set. Re-run **Predictors** with **↻ Predictors / Enriched Timeline** ON for the affected test repos so train and test share the same columns.

### Validation experiments (optional)
The k-fold developer CV, leave-one-project-out, horizon ablation, and predictor-importance runs are not buttons. They run from the command line and read the same `repo_split.csv`. If all the data is ok run:

    python "Dashboard/DemoAppV2.2.py" --eval-suite

Adjust the search space by editing the hyperparameter section in the script. This can take many days ;)


## Step 4 — Open the DISTRAC app

Hopefully we have gotten to this step Ok and just need to run DISTRAC, it is a standalone tool with its own FastAPI backend. We do not need to provide anything to this app by hand as everthing should be provided from the pipeline. It reads the predictions written in Step 3 and serves an interactive dashboard the app explains itself once it's open. Hopefully you have seen a video of its use or read the paper and can see more details there.

### Run
1. Start the server: `python Extractors/distrac_api.py`
2. Open **http://localhost:8000** in a browser.
3. Pick a repository from the selector to view its developers, activity, and inactivity-risk
  predictions.

## End of Replication

This is the end of the replication portion of this project, and now I will go more in-depth on development and ideas that support it. 

# Context

You can use AI to summarise this, as it is quite long, but if you can spare the attention span, I think I wrote something worth reading fully:

This project started with somebody else's tool and not with a machine-learning question.

A research group built a program that could look at an open-source developer's history on GitHub and automatically label when that person had gone inactive. That was the new capability, and it's the thing everything here is built on top of. Igor read that work and saw he had an idea: if a programme can *label* past inactivity for us automatically, then maybe we can go one step further and predict future inactivity. That question became the grant that funds this work. And once you can predict something, the next question writes itself, which is, if we can tell that a developer is drifting away, can we build something people actually use that surfaces it? Label, then predict, then app. That's the whole arc, and each layer sits squarely on the one below it.

It's worth being clear about why that's an unusual setup, because it colours everything that follows. In a normal machine-learning project you're handed a clean, labelled dataset. Somebody already wrote down which image is a handwritten seven, which iris is which species, and which census row earns over fifty thousand a year. The labels are given, and you trust them. We don't have that. Nobody ever told us when a developer became inactive; there's no ground-truth column sitting there waiting for us. Instead, the labelling programme manufactures our response data out of raw GitHub activity by applying a set of rules. That manufactured label is the novelty of the project, and, as I'll keep coming back to, it's also its softest spot. So before anything else, you have to understand how that label gets made, because it is the response column for every prediction we make.

## Start

The very first stage of DisTrac was getting that labelling programme working and understanding it inside out. The whole point is that we don't want to hand-label every period of a developer's life, so we use a rule to identify a break automatically. Before you read how it actually works, stop and think about how you would do it not using a computer. If you were the manager of a hundred developers, what rules would you write to decide when someone was on a break versus just working at their own pace?

In a traditional workplace this is easy, because there are timesheets and clock-in times. We strip out all the ambiguity about when someone was active by simply asking them, and as long as we assume everyone's honest, the "perfectly accurate" truth gets handed to us at least if everyone has flawless memory and never fudges a date. But we know that's not how it really goes; people forget, round off, and occasionally lie. In open source we don't even get that flawed luxury. We have to collect everything a developer ever contributed and then label the gaps between those contributions as active or inactive *without ever asking them*. We are reconstructing their activity entirely from their footprints on GitHub. So let's define how a person might reasonably draw the line between active and inactive.

My working definition: "Activity is small, consistent breaks between contributions, and inactivity is when the time between contributions gets longer than normal."

That one sentence has to absorb every edge case and every bit of context, and you can already feel how hard it is. What is "normal", and what counts as a "small consistent break"? The original paper split this logic across two files: one in a `BreaksManager 'folder' and 'BreaksLabeling' in another. We kept a similar structure but refactored it into one file.

The first part is finding the periods that are longer than normal and calling those breaks. The second part is labelling the time inside each break as one of three things: inactive, non-coding, or gone. That three-way split matters, because a developer isn't simply on or off. Someone can be active without writing any code, and someone who hasn't touched the project in a year is in a very different situation than someone who went quiet last week. These labels aren't hard to assign once you know a break happened. "Gone" is the simple one: if we haven't seen any activity in a year, the person isn't *inactive; they're *gone*. "Non-coding" is fuzzier, and honestly, I'm still not a hundred per cent on the exact mechanics; it's something like a rolling window that checks for activity that isn't code, like issues and comments. To me it mattered less because those developers are genuinely *active* in my book. They're still doing something.

The part that's actually interesting to tinker with is how you define a break in the first place, because what we're really doing is hunting for outlier time periods. We collect the list of gaps between a developer's commits, so commit 1 to commit 2 was 2 days, commit 2 to commit 3 was 10 days, commit 3 to commit 4 was 3 days, and so on, and then we look for the abnormally long ones. Crucially, we don't compare against the developer's whole history, because people's rhythms change and we want to judge a gap against its neighbours. If someone commits every day and then disappears for 10 days, that's an outlier, but if that same person years later only commits weekly, a 10-day gap isn't unusual at all. So we use a simple interquartile range (IQR) calculation to set a threshold per developer, where anything longer than the threshold counts as a break. An everyday committer might land on a 5-day threshold, while the weekly committer gets something closer to 30.

That's the heart of it. If you want to go deeper, the code that runs all of this is `label_developers_activity`in the DemoApp, which is the function that calls each piece in order (it starts with `write_pauses_table`), and the original code and paper are here if you ever need the source of truth, though you don't need to read them to keep working:

https://github.com/collab-uniba/developersInactivityAnalysis

https://zenodo.org/records/4590208

https://arxiv.org/abs/2103.04656

Now, the honest reason I've spent this much time is how the label gets made. It's the part of the project I'm least comfortable with, and it's the thing you'll most want to improve. Because we manufacture the label from a rule rather than observing a real event, there's no clock-out, no goodbye message, and no manager flagging someone as at-risk; we have ambiguity about what we're even predicting. And that ambiguity is exactly what limits how actionable the predictions can ever be. If our response column were something concrete, like a timesheet clock-out or the moment a maintainer messaged someone to ask why they'd gone home, the prediction would have a clear thing to act on. As it stands, it doesn't. I'll be straight with you: this gave me enough pause that I questioned whether the work was worth publishing. I'll make the full argument later in the Limitations and Future Work section, and I've come around to thinking it's a more interesting problem than a flaw because it points at something that open source is missing – an "operational layer" that would make any prediction worth acting on. Right now I made a distraction, and it's ok, but for now, just hold onto one idea as you read the rest: everything downstream inherits whatever this label does and doesn't capture. It is the foundation, soft spot and all.

With that in mind, let's look at where the raw material actually comes from.

# Data Collection

Before any of the prediction or labeling can happen, we need data, and all of it comes from GitHub. GitHub has an API, which is basically a way for a program to ask GitHub questions and get answers back instead of you clicking around the website by hand. Warning: the API is confusing and has about a million rules for what you can ask, how often, and what shape the answer comes back in. The good news is that most of that doesn't matter for understanding what we are doing, just the how. The collection program already deals with all of it, and hopefully it won't need to be remade. What you actually need to understand is what we are pulling out of GitHub and why, because that data is the foundation everything else sits on.

I want to come at this in three passes. First I'll explain what we are collecting, because that's the part that matters most. Then I'll talk about where the data falls short and the limitations you need to keep in the back of your head the whole time. And only at the end will I get into the technical machinery, the workers, the scheduling, and the multiple API keys, because honestly, you'll see most of that the moment you run the program yourself.

### Theory

The whole goal is simple to say: we want to know when developers interacted with a project. That's it. Every time someone does something, we want a record of it and a timestamp. On top of that, we grab a few extra things we'll use later when building the DisTrac app, but the core is "who did what and when."

GitHub uses its own vocabulary for the different kinds of things a developer can do, but it really comes down to three: commits, issues, and pull requests (PRs). If you already know these, skip the next few paragraphs. If you don't, please read them, because the entire rest of the project is built on top of these three ideas.

A commit is someone making edits to a project's files and then committing those changes. Think of editing a Google Doc with a team, where everyone is typing in the same document and it updates live for everybody. A repository (the "repo," which is GitHub's word for a project's folder of files) works a little differently: you make your changes on your own machine, and then you have to save them and send them off to the rest of the team. A commit is one of those "save and send" moments. For our purposes, each commit is a signal that the developer finished some chunk of work. It's the clearest "I did something" event we get.

A pull request, or PR, is a developer saying, "Here's a batch of changes I'd like to add to the project. Can someone review it?" It's a proposal. Other people look at it, comment on it, and ask for changes, and eventually it either gets merged into the project or it doesn't. So a PR isn't just one event; it's a creation followed by a whole little conversation that hangs off of it.

An issue is closer to a comment or a ticket on the project. It's where people report bugs, request features, or just discuss things. Like a PR, an issue has a moment where it gets created, and then a stream of activity that happens to it afterward: people comment on it, label it, assign it to someone, link it to some code, close it, reopen it, and so on. GitHub treats every one of those little actions as its own separate event, which is exactly why, as you'll see in a minute, the issue data ends up being the messiest of the three.

Skip to here:

Running the actual collection is genuinely easy. You hand the program two things, a list of repos you want to collect and a list of GitHub API tokens to use, and it goes. The repo list looks like this:

```
Rdatatable/data.table

JabRef/jabref

rails/rails

facebook/react
```

Each line is in the form `organization/repo` "We recently added the ability to also pull in commits from the other repos belonging to the same organization, so the program can reach out to Facebook's other projects too. The tokens are just GitHub API keys, and I'll explain why you want several of them down in the technical part.

When it finishes, the program sorts everything by organization and then by repo so nothing gets mixed up. For `facebook/react`, the folder ends up looking like this:

```
Organizations/

  facebook/

    react/

      commit_list.csv

      per_file_commits.csv

      issues.csv

      issue_activity.csv

      prs_repo.csv

      prs_comments.csv

      repo_tree.csv

      pauses_commits.csv
```

Two files for commits, two for issues, two for PRs, and one for the repo's file structure, which, if you're counting, is seven raw collection files. Let me walk through them in plain terms. `commit_list.csv` is the list of commits to the repo, one row per commit. `per_file_commits.csv` is that same activity but broken down file by file, which you'll want when you do deeper analysis and care about which parts of the codebase someone was touching. `issues.csv` holds issue creations, one row each time an issue is opened. "Events" is where all those little issue events I mentioned land: comments, subscriptions, mentions, labels, closings, cross-references, renames, assignments, and a couple dozen more types. `prs_repo.csv" Holds PR creations and mostly holds the comments and discussion on them. Finally, it is just the structure of the repo's files and folders.

### Technical Aspects 

Okay, the machinery. I'll keep this part lighter, both because you'll watch it happen when you run the program and because at the end of the day all of it exists just to solve one annoying problem: getting a lot of activity out of GitHub without GitHub cutting you off. The main program you run is "", and that's the main file that kicks everything off. The reason for the multiple API tokens is that GitHub limits how many requests a single token can make in an hour. One token trying to collect a big project like React would take forever. So the program runs several workers in parallel, each one using a different token, which multiplies how fast you can pull data. To keep those workers from stepping on each other and collecting the same thing twice, there's a scheduler, "Scheduler," whose entire job is to hand out the work and let each worker "claim" a page of results, so two workers never grab the same page. While the collection is running, you'll get a progress bar not for the scheduler, so you can see how far along it is.

There's a third file worth knowing about. I wrote it to grab developers' profile pictures and bio information so the DisTrac app could show something nicer than a bare username. You don't need it for the analysis at all, only for the app. There are a couple more files sitting in the Data Collection folder that are old and can be looked at to see if we missed something like non-merged commits.

Pick a repo, ideally a smaller one so it doesn't run all night, and collect it. Then open up the files. I think looking at `issue_activity.csv` and finding the unique values in the "event-type" column is useful to see what we are doing. You'll see the whole zoo of GitHub event types show up, things like IssueComment, LabeledEvent, ClosedEvent, CrossReferencedEvent, and on and on. Load the file into a dataframe (Pandas is plenty), poke at the unique values in a few columns, and just get a feel for what one row actually looks like across each of the files. Once you've seen the real data with your own eyes, everything I said above will click into place in a way it won't from reading alone. Seriously, go run it before you move on to the analysis.

# Analysis (Step 2: Process the data into a model-ready table)

Everything in this section is orchestrated. Early on, when I was trying to get all the moving pieces to play together, I built a small Streamlit app that lets you keep a list of repos and then load, process, or overwrite the data for each one. There are a lot of little steps and a lot of scheduling involved, and rather than run them by hand every time, I wrapped them behind buttons. The one that matters most is the Predictors button: when you click it, the app walks every repo in your list through the whole pipeline, from the raw CSVs all the way to a single table that's ready for modeling. 

Three jobs: structuring the data, building the response, and building the predictors.

### Structuring the data

The first job is structuring. Raw collection leaves us with a pile of separate event tables, commits, per-file commits, issues, issue activity, and PRs and PR comments, and a model can't do anything with that directly. We need one tidy table where each row is a single developer on a single day. So the first thing the pipeline does (`load_users_activity`) is read all those raw tables back in for a repo. But before it builds anything, it has to decide which developers are even worth looking at, because a big project has thousands of one-time contributors, and we don't care about predicting when a drive-by typo-fixer "leaves." That's what Truck Factor is for.

The truck factor is the one piece of structuring that needs a real explanation. The name comes from a slightly morbid thought experiment: how many developers would have to get hit by a truck, that is, leave the project, before it's in serious trouble because too much knowledge walked out the door with them? A project with a truck factor of 2 is fragile, since two people hold most of the knowledge, while a project with a truck factor of 20 is resilient. We don't use it here to measure fragility, though; we use it to select the core developers, the "heroes," who actually carry the project. Those are the people whose timelines we build and whose disengagement we care about predicting. The calculation lives in the `kd` module ("") and it leans on a notion called "Degree of Expertise," which is roughly how much of a given file's knowledge a developer "owns" based on who's authored and edited it over time. If you ever need to defend or change how we pick the core set, the real grounding is in the literature: Truck Factor traces back to Avelino et al., and the Degree of Expertise / code-ownership idea traces back to Bird et al. Their papers are the place to go, not this document.

Once we know who the core developers are, the pipeline builds the timeline (`timeline(...)`). This is the transformation that takes individual time-stamped events and rolls them up into one row per developer per day, counting how much of each kind of activity happened. So this:

```

1/2/2024, 12:40am, commit, Dev A

1/2/2024, 12:40am, issue,  Dev A

```

becomes this:

```

1/2/2024, Dev A, 1 commit, 1 issue, 0 prs

```

That daily, per-developer timeline is the spine of the entire project. Everything else, the response and every single predictor, gets attached to it as extra columns. If you only remember one artifact from this whole section, remember that the unit we work in is the developer-day.

### building the response

The second job is building the response, the thing we're actually trying to predict. This is the labeling step, and it adds a `state` column to the timeline saying what the developer was doing in that period: active, or one of the flavors of inactive we covered back in the Start section (inactive, non-coding, or gone). The actual logic for how a break gets found and labeled is genuinely involved; it's the per-developer IQR-threshold business from earlier, so I'm deliberately not re-explaining it here. If you want the theory, go back to the Start section, and if you want the original source, the Collab-Uniba paper linked in the collection section is it. For this section it's enough to know that labeling is where the response column is born. 

### building the predictors

The third job, and the one with by far the most surface area, is building the predictors, all the extra columns we bolt onto each developer day that might help a model see disengagement coming. ALL of these are found from other research. If you want to know about sociotechnical networks, ask Pedor. If you want to read about knowledge islands, read the paper about them. I'm using their code and ideas here. 

All of these have not been checked for cross-correlation, real predictive contribution, or redundancy. They're a generous first draft of "things that might matter." Validating them, pruning the dead weight, and improving the rest is wide-open work, but it is not the bottleneck of the project, so don't feel like you have to perfect it before moving on.

The social-technical network features (the `stn` module) are about who a developer is talking to. From the issue and PR discussions we build a graph of who interacts with whom, and then for each developer-day we count things like how many people they interacted with, how many of those were brand-new contacts, how many were newcomers to the community versus established regulars, whether they committed alone that day, and so on (`issue_interactions_today`, `total_unique_partners_today`, `new_to_community_today`, `regulars_today`, and a dozen relatives). The intuition is that someone quietly withdrawing from the social fabric of a project might be an early warning sign of leaving it. This is the same network you can see drawn in the app's social-technical panel, so when a number looks off, it genuinely helps to go look at the graph. We make interactions with Pedro's idea that people in the same issue or PR have relationships or interactions.

The knowledge-distribution features (back in the `kd` module) are about what a developer worked on rather than who with. For each day we look at which files they touched and whether they were working alone on files they "own" or collaborating on shared ones (`files_worked_today`, `owned_files_today`, `collab_files_today`, `collab_commit_ratio`). When making the truck factor, we have to look at who owns the files, and using some ideas from the same researcher, we determine the degree of expertise (DOE). This DOE is really important! Look how it's made. Everything from that is just categorizing files from the developers' DOE.

The project-health features plus the `project_health.json` step show how active the whole project has been lately, with rolling seven-day totals of commits, PRs, issues, and active developers and friends. A developer slowing down looks very different in a booming project than in a dying one. This can be expanded on. 

Finally, there's a broad expanded set computed all at once that rounds out the picture with the more mechanical features. These are the lagged-activity rolling averages and standard deviations (their own commit counts over the last 7, 30, and 90 days; lifetime totals); their personal break history (days since their last break, how long it was, and how many breaks in the last 90 and 365 days); code churn (lines added and deleted, today and on a rolling average); their tenure on the project in days; a handful of cross-repo features asking whether they've been active elsewhere in the organization while quiet here (`org_active_elsewhere_7d` and similar); and the cyclic time-of-week and time-of-year encodings (`dow_sin`, `dow_cos`, `month_sin`, `month_cos`). If I say "the share of the repo's commit volume this developer accounts for" or "the number of newcomers they talked to," you can already picture the calculation. They're the kind of feature you add because it's cheap and plausibly useful and then prune later once you've actually measured what carries a signal. The cross-repo features come with one setup gotcha worth flagging now: they depend on a ".gitattributes" file sitting at the organization level.

All of those families get computed separately and then merged back onto the labeled timeline by developer and date (the project-health ones by date alone). The result gets saved as one row per developer per day, with the `state` response and every predictor attached. That file is the real output of this whole section. There's also a last step ``that writes a set of parquet files specifically for the DisTrac app to read.

# Machine learning (Step 3 - Train the model and generate predictions)

Everything before this collection, labeling, and the feature pipeline is solid, but the modeling is made on top of it, which is where I was learning as a student. You can see that because of several response formulations (causal and non-causal versions, binary and four-class), a hyperparameter search, and half-finished experiments. Treat this section as "here's what it does and why, and here's where I'd push on it," not "here's the

final answer."

### What the model is actually doing (the 90-day window)

The prediction task is this: stand on some day, look back over that developer's last 90 days, and guess what their state will be a set number of days in the future (you pick the horizon—7, 14, 21, or 30).

Disengagement is a shape over time, not a single day. Two developers can look identical today; both committed once, and both closed an issue, but one has been quietly fading for two months, and the other is rock-steady. That's what an LSTM is for; it's a model that reads a sequence one day at a time and

carries a running "memory" of what it's seen, updating that memory at each step. By the time it reaches the end of the 90-day window, it has compressed the whole trajectory, the trend, the rhythm, and the recent dip into a summary it uses to make the call. The closest everyday analogy is a coach flipping through an athlete's last three months of training logs; they're reading the story and keeping the important bits in their head as they train.

### The look-ahead trap and the causal state column

Here's the subtle thing that cost me the most time, and you need to understand it. Remember how a break gets labeled: the algorithm decides.

whether a gap is an outlier by looking at the gaps around it and it looks both behind and ahead. That's completely fine for describing the past, which is why the `state "column" is good as the thing we predict, the response. Hindsight is allowed when you're labeling.

It is not allowed when you're predicting. If you feed a developer's current state in as a predictor, you add information about the future. because that current-state label was itself computed using days that, from the model's point of view, haven't happened yet. I suspected this was happening, and when I checked, it was. 

The fix is a second, causal version of the state, computed using only a window behind each point (180 days back, no forward peek). That's the version that's safe to use as a predictor. So you have two columns, and you must keep them straight: `state` "bidirectional" is the response, and "state-causal" and "backward-only" are the predictors. Mix them and the leak comes right back.

One honest caveat I never fully closed: even the causal version isn't perfectly clean because of when a break can be labeled at all. You can only know a break happened once it's over, so labeling a just-ended break still leans a little on how it ended. Trying to strip out forward-looking information changed when we're allowed to assign the label, and a small leak may survive that. I'm flagging it, not claiming it's solved. It's the kind of thing that fixes itself when DISTRAC runs live in production; you label events as they actually arrive, so the timing is honest, but for the offline model, know it's there.

### Always test it against a baseline model

The one piece of advice every ML person gave me when doing this project is that I need to pit the fancy model against a stupid one. The stupid model is the bar. If the LSTM can't clearly beat it, you've thrown away interpretability you didn't need to lose.

A natural baseline "model" here would be a flat rule like "if a developer has been inactive for 7 days, predict a break." But there's a trap, and it's the same leak as above: a baseline built on the bidirectional `state "will look unfairly strong" because that state already peeked ahead; you'd be racing the LSTM against a cheating baseline. The fair comparison uses the causal state.

# DisTrac (Step 4)

This is the final piece of the project, and I think of it as a first attempt at building a real tool for open source maintainers. We have a solid structure with two pages. One shows the model's predictions and context about developers; the other lets you simulate what happens when someone leaves. What needs work is the visualization, how we present information, and which information we're actually showing. Both pages sit on top of the pipeline that came before them, but they're trying to solve two very different problems.

## The Two Pages

Page 1 shows you the repository you're working with and the model's output about each developer. It's the "what does the data tell us right now" page.

Page 2 lets you simulate a developer's departure and see what breaks. It's the "what would happen if this person left" page.

These pages are fundamentally different in how they got built. Page 1 is trying to visualize something we don't have a real analog for yet, so the design is partly invented. Page 2 is different. It draws on established methods from open-source research to show dashboard-style metrics. The key difference is that Page 1 requires us to create ideas about what matters, while Page 2 lets us find those ideas in existing literature and adapt them.

## Page 1: Model Output and Developer Context

Page 1 is where maintainers see the repository they're interested in and learn about the people who work on it. This is where the model's predictions live, but more importantly, it's where you get context about what those predictions actually mean. A high disengagement score for a developer only matters if you understand who they are, what they work on, and how critical their work is to the project.

## Page 2: Developer Departure Simulation

Page 2 is built around three dimensions of project health that matter when a developer leaves. The original project vision outlined these clearly:

1. "The first dimension examines activity drops through statistical analysis of historical break patterns and their effects."
2. "The second dimension focuses on knowledge distribution and technical risk."
3. "The third dimension examines community health impacts through sociotechnical network analysis."

To implement these three dimensions, I built on existing research. I used Perdor's work for the sociotechnical network. I used Truck Factor code for knowledge distribution and technical risk. Lastly, I tried using pre-tests and post-tests but landed on simple CHOSS metrics for historical break patterns.

Definition:

"We develop an interactive tool called DISTRAC (DISENGAGEMENT TRACKER) to aid OSS project maintainers by providing real-time predictive insights into potential contributor disengagement. Building on the prediction model established in Thrust 1, DISTRAC will empower maintainers to proactively manage community dynamics and continuity by anticipating the likelihood and impact of contributor leaves. DISTRAC’s key features include predictive displays, impact assessment, simulation capabilities, and dependency mapping, each designed to address maintainer needs for efficient community management and resilience. 

The tracker will highlight contributors’ likelihood of disengaging (taking a break or leaving), enabling maintainers to see how the project may be at risk. Alongside this, we will provide an impact assessment functionality to simulate the consequences of a contributor’s absence, according to metrics such as committed lines of code (LoC), pull request activity, issue interactions, module or file ownership, and the contributor’s expertise with APIs. This analysis will be instrumental in identifying critical dependencies and assessing the potential impact on the project’s stability and progress. The simulation tool will allow maintainers to manually input hypothetical disengagement for specific contributors, enabling them to understand the potential impact of such scenarios. This feature is useful, for example, when contributors informally communicate planned absences, allowing maintainers to prepare for possible disruptions. "This said true to the application's goal, but I don't know if it was ever fully activated/completed.

# Limitations and Future Work

If you understand the pipeline now and you're thinking about what to improve, this section matters. Still, the overarching idea will still be to make a dashboard or use data to help combat developer inactivity in OSS projects. We have two paths we can follow.

## The Core Problem

The entire project sits on top of one decision: we manufactured the label we're trying to predict. We looked at gaps in a developer's GitHub activity and said, "Gaps bigger than normal count as breaks." Then we labeled the time inside each break as inactive, non-coding, or gone. This rule is useful for describing the past. It lets us go back and say, "Yeah, this person was quiet from January to March." Hindsight is allowed when you're describing what already happened.

But the model uses this manufactured label as its response. A perfect weather forecast tells you something concrete: it will rain tomorrow, so bring an umbrella. The forecast is useful because rain is a real thing in the world, and the action is a jacket. With our model, the maintainer sees high disengagement risk and has to ask themselves, "What am I supposed to do with this?" Should I reach out to the person before they leave? Should I leave them alone? Should I start planning for someone else to take over their work? The thing being forecast isn't a concrete event in the world, so the action you take is unclear, and the consequences of acting are unpredictable. Some actions we could ask a user to make have high social barriers and stigmas we would need to cross, and this makes people uncomfortable and unlikely to use the tool. 

## Two Paths Forward

There are two ways to address this. The first is to fix the prediction itself. The second is to build a better tool that doesn't rely on prediction at all.

If we go the prediction route, we need to predict something concrete. A few ideas:

Survival framing reframes the question from "will they break" to "what's their risk over time." Instead of a binary yes-or-no, you get a developer's survival probability day by day and watch for when it drops. I think this can be more honest about uncertainty and more actionable.

Or predicting break length shifts the timing. Instead of asking "when will they disengage," ask "how long will they stay?" Predict the length of the next break as a number. This gives a maintainer lead time tied to a real moment: we predict someone will take a two-month break starting next week. Break-length regression drops classification entirely and just predicts the duration of the next inactive period as a continuous number.

But there's a second path that I've come to think might be more valuable: build a tool that doesn't predict anything at all.

The coolest part of this project, I've been told, is the developer activity tracking. Not the predictions, but the visibility. Showing a maintainer exactly what's happening right now, what the past looks like, what they work on, and how the team's energy flows. A red-yellow-green activity status tells you someone is away. Showing their social network, their code ownership, and their contribution style tells you what you'd lose if they didn't come back. You don't need a complex prediction to make this useful.

This idea of an application that helps with scheduling, coordination, and understanding team dynamics is interesting. It solves real problems in open source. You don't know if your core person is busy or gone; you don't know who else understands their code; you don't know if the quiet period is temporary or permanent. An honest view of the present state answers all of those questions without needing to forecast anything.

Maybe the prediction we actually need comes after we build something useful. Once maintainers are using the tool, once we can see what questions they actually ask and what decisions they need to make, then we can work backward to the prediction that matters. Maybe it's predicting how long someone will be active or how long a break will last. Maybe it's predicting when activity will resume. Maybe it's predicting which pieces of code will become bottlenecks while someone's away. Those are predictions tied to actionable decisions, not to a rule we made up on the spot.

The throughline for all of this is the same: stop predicting labels that don't correspond to anything and start building tools that solve the actual problems maintainers face.