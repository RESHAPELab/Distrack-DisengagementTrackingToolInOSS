# DisTrac

This is the final piece of the project, and I think of it as a first attempt at building a real tool for open source maintainers. We have a solid structure with two pages, each serving a different purpose: one shows the model's predictions and context about developers, the other lets you simulate what happens when someone leaves. What needs work is the visualization, how we present information, and which information we're actually showing. Both pages sit on top of the pipeline that came before them, but they're trying to solve two very different problems.

## The Two Pages

Page 1 shows you the repository you're working with and the model's output about each developer. It's the "what does the data tell us right now" page.

Page 2 lets you simulate a developer's departure and see what breaks. It's the "what would happen if this person left" page.

These pages are fundamentally different in how they got built. Page 1 is trying to visualize something we don't have a real analog for yet, so the design is partly invented. Page 2 is different. It draws on established methods from open source research to show dashboard-style metrics. The key difference is that Page 1 requires us to create ideas about what matters, while Page 2 lets us find those ideas in existing literature and adapt them.

## Page 1: Model Output and Developer Context

Page 1 is where maintainers see the repository they're interested in and learn about the people who work on it. This is where the model's predictions live, but more importantly, it's where you get context about what those predictions actually mean. A high disengagement score for a developer only matters if you understand who they are, what they work on, and how critical their work is to the project.

## Page 2: Developer Departure Simulation

Page 2 is built around three dimensions of project health that matter when a developer leaves. The original project vision outlined these clearly: activity impact, knowledge risk, and community health. Each one asks a different question about what breaks when someone goes.

The first dimension tracks what happens to overall activity. When a developer steps away, the project's pace of work drops, issues go unanswered longer, and pull requests stack up. Understanding this pattern historically tells you what you can expect if it happens again.

The second dimension is about knowledge concentration and technical risk. If the person who left owned the core files or was the only one who understood a particular system, the project is in trouble. This is where truck factor and code ownership concepts come in. Some developers are critical points of failure; others are useful but replaceable.

The third dimension is about the social structure of the project. Developers don't work in isolation; they collaborate, review each other's work, help newcomers. When someone leaves, you lose those relationships and the knowledge transfer that happens through them. The social network analysis shows you how connected each person is and what roles they play.

To implement these three dimensions, I built on existing research. For the social network piece, I used methods from network analysis of open source projects. For knowledge distribution and risk, truck factor and code ownership calculations tell you who holds what expertise. For activity impact, I looked at break patterns in project history and used standard metrics to measure the disruption.

## What Still Needs Work

The app structure is there, but the presentation is rough. The visualization choices are functional but not intuitive, the information density is wrong in places, and we're not always showing the most useful view of the data. That's the honest assessment. It works, and a maintainer could use it, but it doesn't yet feel like a tool that was designed with them in mind.

---

# Limitations and Future Work

If you understand the pipeline now and you're thinking about what to improve, this section matters. It's not a list of what went wrong; it's about a more fundamental choice about what we're actually trying to do.

## The Core Problem

The entire project sits on top of one decision: we manufactured the label we're trying to predict. We didn't observe it in the world; we invented it using a rule.

Here's what that means. We looked at gaps in a developer's GitHub activity and said, "gaps bigger than normal count as breaks." Then we labeled the time inside each break as inactive, non-coding, or gone. This rule is useful for describing the past. It lets us go back and say, "Yeah, this person was quiet from January to March." Hindsight is allowed when you're describing what already happened.

But the model uses this manufactured label as its response. It tries to predict it. And that's where the problem lives.

A perfect weather forecast tells you something concrete: it will rain tomorrow, so bring an umbrella. The forecast is useful because rain is a real thing in the world. With our model, the maintainer sees high disengagement risk and has to ask themselves: what am I supposed to do with this? Should I reach out to the person before they leave? Should I leave them alone? Should I start planning for someone else to take over their work? The thing being forecast isn't a concrete event in the world, so the action you take is unclear, and the consequences of acting are unpredictable.

This isn't a flaw I can fix with better data or a fancier model. Even a perfect LSTM trained on perfect data would still be predicting a rule's output, not a real-world happening. The ceiling on how useful the predictions can be is already in place. We're not forecasting behavior; we're forecasting whether the data matches our rule.

## Two Paths Forward

There are two ways to address this. The first is to fix the prediction itself. The second is to build a better tool that doesn't rely on prediction at all.

If we go the prediction route, we need to predict something concrete. A few ideas:

Survival framing reframes the question from "will they break" to "what's their risk over time." Instead of a binary yes-or-no, you get a developer's survival probability day by day and watch for when it drops. This is more honest about uncertainty and more naturally actionable.

Predicting break length shifts the timing. Instead of asking "when will they disengage," ask "how long will they stay?" Predict the length of the next break as a number. This gives a maintainer lead time tied to a real moment: we predict someone will take a two-month break starting next week.

Break-length regression drops classification entirely and just predicts the duration of the next inactive period as a continuous number.

But there's a second path that I've come to think might be more valuable: build a tool that doesn't predict anything at all.

The coolest part of this project, I've been told, is the developer activity tracking. Not the predictions, but the visibility. Showing a maintainer exactly what's happening right now, where people are quiet, what they work on, and how the team's energy flows. A red-yellow-green activity status tells you someone is away. Showing their social network, their code ownership, and their contribution style tells you what you'd lose if they didn't come back. You don't need a complex prediction to make this useful.

This idea of an application that helps with scheduling, coordination, and understanding team dynamics is genuinely interesting. It solves real problems in open source: you don't know if your core person is busy or gone; you don't know who else understands their code; you don't know if the quiet period is temporary or permanent. An honest view of the present state answers all of those questions without needing to forecast anything.

Maybe the prediction we actually need comes after we build something useful. Once maintainers are using the tool, once we can see what questions they actually ask and what decisions they need to make, then we can work backward to the prediction that matters. Maybe it's predicting how long a break will last. Maybe it's predicting when activity will resume. Maybe it's predicting which pieces of code will become bottlenecks while someone's away. Those are predictions tied to actionable decisions, not to a rule we made up on the spot.

The throughline for all of this is the same: stop predicting labels that don't correspond to anything in the world, and start building tools that solve the actual problems maintainers face.
