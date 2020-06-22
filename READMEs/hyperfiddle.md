# hyperfiddle.cljc — a Hypermedia Function <http://www.hyperfiddle.net/>

Hyperfiddle isolates your web clients from I/O, so your code can stay pure. If React.js is managed DOM,
 Hyperfiddle is managed network and database. This enables a new kind of composable primitive for constructing web software, with paradigm-changing implications.

## Dependency coordinates — Todo

    [com.hyperfiddle/hyperfiddle "0.0.0"]

## Motivation

**The hardest part of web dev is I/O:** data sync between database, various services, UI and then back to the database. An async, slow, failure-prone concern wired throughout the full stack, and the reason why we all code the same web boilerplate over and over again, year after year.

Hyperfiddle makes data-sync invisible with immutability:

* Fiddle graph captures service inter-dependencies as data (query X depends on queries Y and Z)
* Fiddle graph captures application essense (API, UI, database) as one concern – no frontend/backend dichotemy
* Optimizing I/O runtime – data sync reduces to a graph partitioning problem
* Managed data sync – Userland is not concerned with effects, async, errors or latency
* Transport layer independence – swap transport strategies (e.g. REST, websocket) without changing your app
* Dynamic transport strategies can automatically balance caching and latency
* Platform independence – run on any platform (e.g. browser, Node, mobile) without changing your app

**Framework or library?** Neither: Hyperfiddle is server infrastructure, like Apache or Nginx. There is a Clojure library for making custom servers (e.g. integrations and control over data sync) and a client library for talking to it.

## How does it work?

The Cognitect stack has the right primitives to build "functional" data sync by orienting the entire stack
around values and immutability.

* Clojure/CLJC - Value-oriented programming on any platform
* EDN - Extensible notation for values
* Transit - Protocol for value interchange between foreign platforms
* Datomic - Database as a value
* (supplemented with React.js/Reagent)

Hyperfiddle uses the Cognitect stack as a basis to abstract over client/server data sync for APIs, by extending
Datomic's immutable semantics to the API. Unlike REST/GraphQL/whatever, Hyperfiddle's data sync *composes*.

Managed I/O means, as a web dev, you are no longer concerned with remote data fetching or coding HTTP backends.
In fact there is hardly any "web programming" left at all. But managed I/O is not the point. The point is:
*what does managed I/O make possible that wasn't possible before?*
