---
url: https://chatgpt.com/c/6a7b636d-4ab8-83eb-a2bd-2b29a354947a
source: chatgpt
kind: deep_research
saved: 2026-08-11T18:15:34.856056+00:00
---

# Rešerše prior work: goal-conditioned distance jako akumulovaná délka latentní trajektorie iterativní sítě

Research completed in 9m · 
 citations · 
 searches
Rešerše prior work: goal-conditioned distance jako akumulovaná délka latentní trajektorie iterativní sítě
Rešerše prior work: goal-conditioned distance jako akumulovaná délka latentní trajektorie iterativní sítě
Exekutivní shrnutí

Hlavní závěr: v literatuře, kterou jsem prošel napříč goal-conditioned RL, quasimetric learning, recurrent/iterative reasoning, neural algorithmic reasoning, latentní Riemannovskou geometrií, neural ODE, Eikonal/PDE metodami a computational neuroscience, jsem nenašel práci, která by spojovala všechny klíčové prvky vašeho readoutu:

[ \hat d_\theta(s,g)

\alpha_\theta \sum_{t=0}^{T-1} \sum_i \left| z_i^{(t+1)}-z_i^{(t)} \right|2, \qquad \alpha\theta>0, ]

kde (z^{(t+1)}) vzniká opakovanou aplikací stejného bloku na jednu goal-conditioned inference instanci a tato diskrétní délka vlastní skryté inference trajektorie sítě je přímo predikovanou shortest-path hodnotou.

Nejbližší literatura se štěpí do tří téměř komplementárních větví:

Bansal et al., End-to-end Algorithm Synthesis with Recurrent Networks: Logical Extrapolation Without Overthinking, NeurIPS 2022 mají velmi podobnou weight-shared iterativní architekturu a „recall“ původního vstupu při každé iteraci. Jejich odpověď je ale čtena klasickým headem z aktuálního/final hidden state; nic jako (\sum_t|\Delta z_t|) se do výstupu nesčítá. Navíc extrapolaci na těžší úlohy dosahují typicky větším počtem inference iterací, tedy opačně než váš fixed-(T), magnitude-carrying mechanism. 
1

Riemannovské latentní geodesiky, zejména Arvanitidis, Hansen & Hauberg, Latent Space Oddity, ICLR 2018, skutečně definují vzdálenost jako integrál lokální rychlosti podél latentní křivky. To je matematicky téměř přesně princip „distance = path length“, avšak křivka je geodesika v latentním prostoru modelu, získaná samostatnou optimalizací/geodesic solve; není to skrytá trajektorie recurrent inference sítě a nejde o goal-conditioned value learner. 
2

V práci o geometrii hidden-state trajektorií už existuje přesný objekt (\sum_t|h_{t+1}-h_t|). Hasani et al., Liquid Time-constant Networks, AAAI 2021 používají arc length latentních dynamik jako míru expresivity, nikoli jako predikovaný target; ještě explicitněji velmi recentní preprint Truth as a Trajectory z března 2026 definuje layerwise arc length přesně jako (\sum_\ell|h_{\ell+1}-h_\ell|_2), opět pouze jako diagnostickou statistiku reprezentací. 
3

To vede k poměrně silnému, ale úzce formulovanému novelty claimu:

Z dostupného prior work se jeví jako nový krok použít arc length / total variation vlastní fixed-depth recurrent latent inference trajectory přímo jako parametrizaci goal-conditioned shortest-path value function, přičemž lokální latentní displacementy nesou velikost vzdálenosti a jejich normy jsou jediným distance readoutem.

Naopak by nebylo bezpečné tvrdit obecně, že je nové „chápat vzdálenost jako path integral“, „opakovaně vkládat vstup do recurrent network“ nebo „akumulovat lokální náklady“: každá z těchto idejí má jasné předchůdce. 
4

Za zvlášť silnou a podle nalezené literatury méně anticipovanou vlastnost považuji fixed-(T) horizon extrapolation: obtížnější/odstálenější dvojice nepotřebuje více výpočetních kroků, protože vzdálenost může růst přes velikost každého (|\Delta z_t|). Literatura „deep thinking“ obvykle dělá pravý opak: extrapoluje tak, že při inference přidává recurrences. Samostatná literatura o horizon generalization v RL sice výslovně studuje trénink na blízkých cílech a generalizaci na vzdálené, ale používá jiný mechanismus založený na goal-conditioned policy/value struktuře, nikoli latentní arc-length readout. 
5

Moje celkové posouzení: vysoká pravděpodobnost architektonické novosti této konkrétní kombinace; střední až vysoká pravděpodobnost novosti samotného „latent-inference arc length as value readout“. Absenci samozřejmě nelze rešerší absolutně dokázat a toto není patentová novelty opinion; nicméně v hlavních bezprostředně relevantních liniích není exact match.

Co přesně počítám za shodu

Je důležité oddělit několik matematicky podobných, ale mechanicky velmi odlišných věcí. U vás je recurrent depth (t) interní výpočetní čas, nikoli čas prostředí:

[ z^{(t+1)}

F_\theta \bigl(z^{(t)},E_\theta(s),E_\theta(g)\bigr), ]

s re-injekcí endpointových informací při každé iteraci, a výstup je total variation/arc length

[ L_z

\sum_t\sum_i |\Delta z_i^{(t)}|_2. ]

To není totéž jako běžný residual network identity

[ z^{(T)}-z^{(0)}

\sum_t \Delta z^{(t)}, ]

protože norma se u vás bere před sumací; latentní zatáčky se tedy nevyruší. V kontinuální limitě jde přirozeně o

[ L[z]

\int_0^\tau |\dot z(u)|,du, ]

což je přesně arc length inference flow.

Pro hodnocení podobnosti používám následující hranici. Za „ano“ v sloupci akumulace počítám pouze případ, kdy skalární quantity, která má význam výsledku nebo jeho přímé komponenty, skutečně obsahuje součet/integrál lokálních příspěvků přes interní computational trajectory. Nestačí, že se recurrent state iterativně aktualizuje, že Bellman equation sčítá environment rewards, že ACT sčítá halting probabilities nebo že algoritmus nakonec našel cestu, jejíž edge costs se dají sečíst.

Skóre 5/5 by znamenalo téměř stejnou metodu; 4/5 sdílení klíčového mechanického principu nebo několika velmi specifických architektonických prvků; 3/5 silný matematický či architektonický analog; 2/5 stejný problém, ale jiný mechanismus, případně stejná myšlenka ve vzdálené doméně; 1/5 pouze kontextová souvislost.

Ještě jeden podstatný rozdíl vůči quasimetric literatuře: z vaší konstrukce automaticky plyne nezápornost readoutu, ale sama o sobě nezaručuje triangle inequality mezi třemi různými state-goal queries, protože každá dvojice ((s,g)) může vyvolat jiný podmíněný latentní flow. To je důležitý rozdíl proti PQE/IQE/DeepNorm, kde je triangle inequality součástí konstrukce latentního distance function. Toto je inference z vaší formule a z konstrukcí těchto quasimetric modelů. 
6

Goal-conditioned distance, quasimetrics a temporal representations

Tady je výsledek nejjednoznačnější: relevantní GCRL distance literature téměř bez výjimky čte vzdálenost z dvojice endpoint representations nebo ze scalar value headu, nikoli z dráhy, kterou během inference urazil hidden state.

Práce	Co počítá a jak se distance čte	Akumuluje (\Delta z) přes compute steps?	Goal-conditioned?	Local → long-range extrapolace?	Blízkost
Wang, Torralba, Isola & Zhang, “Optimal Goal-Reaching Reinforcement Learning via Quasimetric Learning”, ICML 2023 
7
	Optimal goal-reaching value je reprezentována quasimetric distance mezi learned representations; distance je pairwise quasimetric readout.	Ne. Žádná recurrent inference arc length.	Ano	Generalizace je motivací quasimetric struktury, nikoli váš fixed-(T) local-distance experiment.	3/5 – téměř totožná interpretace value-as-distance, zcela jiný readout.
Wang & Isola, “On the Learning and Learnability of Quasimetrics”, ICLR 2022 — PQE 
8
	Poisson Quasimetric Embedding mapuje endpointy do latentní quasimetric struktury a vyhodnocuje jejich asymmetric distance.	Ne	Použitelné v goal reaching; paper zahrnuje offline Q-learning.	Ne v požadovaném smyslu	2.5/5 – silný distance inductive bias, ale pairwise static readout.
Wang & Isola, “Improved Representation of Asymmetrical Distances with Interval Quasimetric Embeddings”, NeurIPS NeurReps 2022 
9
	IQE je explicitní latentní quasimetric (d_{\rm latent}(z_x,z_y)), navržená jako jednodušší/generalizující alternativa PQE, Deep/Wide Norm a MRN.	Ne	Obecně ano/aplikováno na RL distance tasks	Testuje quasimetric generalization, ne recurrent horizon extrapolation.	2.5/5
Pitis, Chan, Jamali & Ba, “An Inductive Bias for Distances: Neural Nets that Respect the Triangle Inequality”, ICLR 2020 — DeepNorm/WideNorm 
10
	Neural norm/distance architecture se zabudovanou triangle inequality; distance vzniká z endpoint features.	Ne	Ne specificky; relevantní jako distance parameterization	Ne	2/5
Liu, Feng, Liu & Stone, “Metric Residual Networks for Sample Efficient Goal-Conditioned Reinforcement Learning”, AAAI 2023 
11
	Rozkládá (-Q(s,a,g)) na součet metric component + asymmetric residual component.	Ne. Jejich „summation“ je součet dvou architektonických funkcí, ne součet přes recurrent compute time.	Ano	Důraz na sample efficiency, ne fixed-(T) distance-horizon extrapolation.	3/5 – velmi relevantní value architecture, ale „sum“ je jiného typu.
Myers, Zheng, Dragan, Levine & Eysenbach, “Learning Temporal Distances: Contrastive Successor Features Can Provide a Metric Structure for Decision-Making”, ICML 2024 
12
	Z contrastive successor features odvozuje temporal distance splňující triangle inequality; distance je funkcí naučených predictive representations.	Ne	Ano, přímo temporal goal distance	Ano v podobě combinatorial „stitching“ generalization, ale bez recurrent inference path.	3/5 – velmi blízký target/generalization objective, jiná mechanika.
Hartikainen, Geng, Haarnoja & Levine, “Dynamical Distance Learning for Semi-Supervised and Unsupervised Skill Discovery”, ICLR 2020 
13
	Učí dynamical distance odpovídající očekávanému počtu kroků mezi stavy a používá ji jako reward.	Ne; pairwise learned predictor	Ano v praktickém smyslu state→goal	Ne váš local-only/fixed-compute extrapolation test	3/5 – target „expected steps to goal“ je velmi blízký, readout není.
Eysenbach, Zhang, Salakhutdinov & Levine, “Contrastive Learning as Goal-Conditioned Reinforcement Learning”, NeurIPS 2022 
14
	Goal-conditioned value je spojena s inner productem contrastively learned state/goal representations.	Ne	Ano	Generalizace ano, ne mechanicky relevantní	2/5
Eysenbach, Salakhutdinov & Levine, “Search on the Replay Buffer: Bridging Planning and Reinforcement Learning”, NeurIPS 2019 
15
	Naučené goal-conditioned values fungují jako edge costs v grafu replay states; graph search poté sčítá lokální edge costs podél plánované cesty.	Částečně, ale mimo síť. Akumulace je v externím planning graphu, nikoli přes hidden-state compute iterations.	Ano	Umožňuje skládání dlouhých cest z lokálnějších predikcí.	3.5/5 – důležitý precedent „long distance = sum local learned costs“, ale suma probíhá v prostředí/searchi.
Schaul, Horgan, Gregor & Silver, “Universal Value Function Approximators”, ICML 2015; Andrychowicz et al., “Hindsight Experience Replay”, NeurIPS 2017 
16
	(V(s,g))/(Q(s,a,g)) je běžný scalar neural output; HER mění způsob tvorby tréninkových goal transitions, nikoli readout architecture.	Ne	Ano	Ne inherentně	1.5/5
Myers, Ji & Eysenbach, “Horizon Generalization in Reinforcement Learning”, ICLR 2025 
17
	Explicitně studuje možnost naučit se na nearby goals a řešit vzdálenější goals; analyzuje planning invariance / horizon generalization.	Ne	Ano	Ano – toto je nejbližší precedent k vašemu evaluačnímu claimu.	3/5 – velmi blízký generalization question, nikoli distance readout.
Bae, Park & Lee, “TLDR: Unsupervised Goal-Conditioned RL via Temporal Distance-Aware Representations”, CoRL 2024 
18
	Učí temporal-distance-aware representations pro goal reaching.	Ne	Ano	Částečně, reprezentace mají zachytit temporal reachability	2.5/5
Co z této větve plyne

Nejrelevantnější rozdíl je velmi čistý. PQE, IQE, DeepNorm/WideNorm, MRN, QRL a contrastive temporal distances mění funkční třídu

[ (s,g)\mapsto d_\theta(s,g) ]

tak, aby měla vhodnou geometrii. Vaše metoda místo toho mění mechanismus, kterým jediný query získá scalar:

[ (s,g) \longrightarrow z^{(0)},z^{(1)},\ldots,z^{(T)} \longrightarrow \sum_t |\Delta z^{(t)}|. ]

V nalezené quasimetric literatuře jsem nenašel variantu, kde by PQE/IQE/MRN distance vznikala postupným refinementem a následným součtem velikostí těchto refinementů. IQE dokonce ve svém srovnání charakterizuje PQE, Deep Norm, Wide Norm a MRN jako latentní quasimetric modely nad endpoint representations, nikoli jako recurrent procedures. 
19

Zároveň je Search on the Replay Buffer konceptuálně důležitější, než se může na první pohled zdát: ukazuje, že long-horizon goal distance lze vytvořit kompozicí lokálních learned costs. Rozhodující rozdíl je, že jeho index sumace běží přes environment/replay-graph path, zatímco u vás přes computational depth interního latentního flow. 
15

Successor-representation a Laplacian přístupy jsou ještě vzdálenější: ukládají predictive occupancy/spectral coordinates, z nichž lze odvozovat state similarity, reachability nebo temporal structure; ani zde není distance scalar definován jako arc length hidden inference. Contrastive successor features jsou nejpřímější moderní bridge mezi SR a explicitní temporal metric. 
20

Iterativní sítě, recurrent reasoning a neural shortest-path algorithms

Zde najdeme velmi podobný výpočetní substrate, avšak téměř vždy platí schéma

[ z^{(T)}\xrightarrow{\text{head}}\hat y, ]

nikoli

[ (z^{(0)},\ldots,z^{(T)}) \xrightarrow{\sum_t c(z^{(t)},z^{(t+1)})} \hat y. ]

Práce	Iterativní mechanismus a readout	Sčítá local increments do outputu?	Goal / shortest path?	Extrapolace	Blízkost
Bansal, Schwarzschild, Borgnia, Emam, Huang, Goldblum & Goldstein, “End-to-end Algorithm Synthesis with Recurrent Networks: Logical Extrapolation Without Overthinking”, NeurIPS 2022 
21
	Weight-shared recurrent block; Recall uchovává explicitní původní problem instance a zpřístupňuje ji při dalších algorithmic steps; standardní forma je (f(x;m)=h(r^m(p(x)))).	Ne. Final/intermediate representation jde do output headu.	Mazes aj., ale ne distance oracle	Ano; 9×9 → 59×59/201×201 atd., typicky pomocí více inference iterations. 
22
	4/5 – nejbližší architektura a recall; readout i scaling principle jsou zásadně jiné.
Schwarzschild et al., “Can You Learn an Algorithm? Generalizing from Easy to Hard Problems with Recurrent Networks”, NeurIPS 2021 
5
	Opakovaně používá recurrent computation naučenou na lehkých úlohách.	Ne	Mazes/prefix sum/chess	Ano, explicitně „thinking for longer“ na těžších instancích.	3/5
Dehghani et al., “Universal Transformers”, ICLR 2019 
23
	Stejná transformation se opakuje přes depth/time; reprezentace tokenů jsou iterativně revidovány, případně s ACT.	Ne; výstup je final/ACT-combined representation	Ne specificky	Variable computation, nikoli váš distance extrapolation mechanism	3/5 – weight-shared transformer je velmi blízký substrate.
Graves, “Adaptive Computation Time for Recurrent Neural Networks”, 2016 
24
	Akumuluje halting probability a kombinuje micro-step outputs pomocí halting weights.	Ne v relevantním smyslu. Něco se přes kroky sčítá, ale je to weighted output/halting mass, nikoli nonnegative movement cost reprezentující target.	Ne	Compute roste podle potřeby	2.5/5
Banino, Balaguer & Blundell, “PonderNet: Learning to Ponder”, 2021 
25
	Predikce existuje při každém computation step a learned halting distribution rozhoduje, kde skončit.	Ne; expected prediction/loss pod halting distribution	Ne	U některých algorithmic tasks extrapolace přes adaptive compute	2/5
Tamar, Wu, Thomas, Levine & Abbeel, “Value Iteration Networks”, NeurIPS 2016 
26
	Diferencovatelně unrolluje value-iteration-like Bellman backups; informace o goal/reward propaguje mapou.	Ne jako hidden displacement. Path costs jsou implicitně skládány Bellmanovou rekurzí; čte se finální value/policy representation.	Ano, planning-to-goal	Větší vzdálenost obvykle vyžaduje dostatek VI propagation steps.	3.5/5 – nejbližší „iterative computation of shortest-path quantity“, ale jiná algebra readoutu.
Lee, Parisotto, Chaplot, Xing & Salakhutdinov, “Gated Path Planning Networks”, ICML 2018 
27
	Recurrent/gated reformulace differentiable path planningu.	Ne	Ano	Iteration depth řídí propagation/receptive field	3/5
Veličković, Ying, Padovano, Hadsell & Blundell, “Neural Execution of Graph Algorithms”, ICLR 2020 
28
	GNN napodobuje jednotlivé kroky BFS/Bellman–Ford; shortest-distance estimates se iterativně relaxují.	Ne. Výsledkem je poslední algorithm state / decoded tentative distance, nikoli suma velikostí změn hidden states.	Ano, explicitně shortest paths/Bellman–Ford	Ano na větší grafy, ale algoritmický compute je svázán s propagací.	3.5/5
Zhu, Zhang, Xhonneux & Tang, “Neural Bellman-Ford Networks”, NeurIPS 2021 
29
	Generalized Bellman–Ford skládá learned edge/path representations: generalized product podél path a generalized sum přes paths.	Ne přes hidden displacement. Je zde algebraická akumulace podél grafových cest.	Source-target conditioned link/path reasoning	Inductive graph generalization	3/5
Bai, Kolter & Koltun, “Deep Equilibrium Models”, NeurIPS 2019, a implicit differentiable planning variants	Weight-tied computation se chápe jako řešení fixed point (z^*=F(z^*,x)); root solver iteruje do konvergence. Pozdější differentiable-planning práce aplikují implicitní diferenciaci i na VIN/GPPN-like planners. 
30
	Ne; relevantní je equilibrium/final state.	Někdy planning	Compute je vázán na convergence tolerance	2.5/5
Recall je skutečný a důležitý precedent

Bansal et al. nejsou jen volně podobní. Jejich motivací recall architecture je explicitně udržet původní problem instance jako referenci pro každý algorithmic step. Pozdější literatura popisuje recall network formálně jako

[ x_{t+1}=f_\theta(x_t,x_0), ]

na rozdíl od autonomous loop

[ x_{t+1}=f_\theta(x_t). ]

To je velmi blízko vašemu opakovanému re-injektování start/goal tokens; tento komponent bych proto neclaimoval jako nový sám o sobě. 
31

Zároveň ale Bansal et al. poskytují výborný kontrast pro váš nejsilnější extrapolační claim. Jejich modely řeší těžší mazes tím, že „myslí déle“: například paper reportuje inference se stovkami až tisíci recurrence steps na podstatně větších problémech. U vás je (T) fixní a například dvojnásobná predicted distance nemusí znamenat dvojnásobný compute; může znamenat přibližně dvojnásobnou accumulated latent speed při stejném (T). 
22

To je kvalitativně jiný inductive bias:

text
Copy
Deep-thinking / Bellman propagation
harder or farther problem
        ↓
more compute iterations
        ↓
final hidden state → head

Váš mechanismus
farther goal
        ↓
stejných T compute iterations
        ↓
větší Σ ||Δz_t||
        ↓
distance

ACT, PonderNet a Universal Transformer jsou proto jen částeční předchůdci. Akumulují určité halting weights/probabilities a někdy kombinují predictions z více depths, ale scalar target není interpretován jako součet lokálních nákladů generovaných pohybem hidden state. 
32

Latentní path integrals, neural ODE a Eikonal distance fields

Toto je mechanicky nejzajímavější část rešerše, protože zde existuje přesně stejná matematika arc length, jen obvykle na špatné „ose“: integruje se fyzická/latent-geometric trajectory, nikoli interní inference trajectory modelu.

Práce	Quantity a mechanický readout	Akumulace přes compute trajectory?	Goal-conditioned?	Local → long?	Blízkost
Arvanitidis, Hansen & Hauberg, “Latent Space Oddity: on the Curvature of Deep Generative Models”, ICLR 2018 
2
	Riemannovská geodesic distance je délka křivky mezi dvěma latentními endpointy, tedy integrál lokální metric speed.	Ano jako geometrický path integral, ne jako network compute readout. Křivka je geodesika, ne sequence recurrent hidden states jedné distance-query sítě.	Endpoint-conditioned matematicky; ne GCRL	Ne	4/5 – nejbližší matematická definice výsledného scalaru.
Arvanitidis, Hauberg & Schölkopf, “Geometrically Enriched Latent Spaces”, AISTATS 2021 
33
	Rozvíjí shortest-path/geodesic computations v naučeném latentním Riemannovském prostoru.	Totéž: integruje délku explicitní latentní curve, ne computational refinement trajectory.	Ne RL	Ne	3.5/5
Finlay, Jacobsen, Nurbekyan & Oberman, “How to Train Your Neural ODE: the World of Jacobian and Kinetic Regularization”, ICML 2020 
34
	Penalizuje dynamiku Neural ODE pomocí kinetic/Jacobian regularization; continuous trajectory má lokální velocity/energy.	Ano, integrální quantity přes ODE trajectory existuje, ale je to training regularizer, nikoli predicted distance.	Ne	Ne	3.5/5 – velmi blízké „integral of latent speed/energy“, ale špatná role quantity.
Hasani, Lechner, Amini, Rus & Grosu, “Liquid Time-constant Networks”, AAAI 2021 
35
	Analyzuje continuous recurrent hidden dynamics pomocí trajectory length / arc length v latentním prostoru jako míry expresivity.	Matematicky ano, ale pouze jako diagnostika/analýza modelu, ne supervised output.	Ne	Ne	4/5 mechanicky, 1/5 taskově; celkem 3.5/5.
Raghu et al., “On the Expressive Power of Deep Neural Networks”, ICML 2017 
36
	Trajectory length používá jako geometric measure expresivity hlubokých sítí.	Ne jako output; jde o diagnostic geometric observable.	Ne	Ne	2.5/5
“Truth as a Trajectory: What Internal Representations …”, arXiv, březen 2026 
37
	Definuje přímo (S=\sum_{\ell=0}^{L-1}|h_{\ell+1}-h_\ell|_2) jako scalar arc length hidden-representation trajectory.	Ano – přesně stejný diskrétní matematický operátor. Ale jde o post-hoc descriptor correct/incorrect LLM trajectories, ne learned output.	Ne	Ne	4/5 mechanicky; 2.5/5 celkově kvůli zcela jiné funkci.
Dynamic optimal transport / flow formulations	Benamou–Brenier-like a learned flow metody minimalizují integrály lokální kinetic/transport cost podél flow. Moderní neural OT/flow-matching práce zachovávají tuto interpretaci nákladů. 
38
	Ano jako objective transport trajectory, ne hidden-inference readout	Typicky endpoint distributions, ne state-goal value	Ne	3/5
Li, Qiu & Calinon, “A Riemannian Take on Distance Fields and Geodesic Flows in Robotics”, arXiv 2024; IJRR 2026 — Neural Eikonal Solver 
39
	PINN řeší Riemannovskou Eikonal PDE; naučené coordinate network přímo vrací continuous geodesic distance field. Start/boundary-conditioned variant dovoluje arbitrary source-target queries; geodesika se poté získá gradient flow.	Ne. Distance je scalar field solution; gradient-flow trajectory není quantity, jejíž hidden displacementy se sečtou do readoutu.	Ano v endpointovém smyslu	Trénink dokonce nevyžaduje distance labels, ale nejde o local-supervision extrapolation	3.5/5
Ni & Qureshi, “NTFields: Neural Time Fields for Physics-Informed Robot Motion Planning”, ICLR 2023 
40
	Neural time field reprezentuje arrival time řešením Eikonal-like physics constraint; trajectory se získá z field gradients.	Ne	Start/goal motion-planning setting	Generalizace přes query space, nikoli váš mechanismus	3/5
Giammarino, Ni & Qureshi, “Physics-informed Value Learner for Offline Goal-Conditioned Reinforcement Learning”, 2025 
41
	Přidává Eikonal physics constraint přímo k goal-conditioned value learning, se zaměřením na geodesic/long-horizon strukturu.	Ne; value je přímo parametrizovaný scalar	Ano	Long-horizon generalization je explicitně relevantní	3.5/5
Giammarino & Qureshi, “Goal Reaching with Eikonal-Constrained Hierarchical Quasimetric Reinforcement Learning”, 2025/ICLR 2026 
42
	Kombinuje QRL/quasimetric distance s Eikonal constraint pro goal reaching.	Ne	Ano	Ano v long-horizon goal reaching sense	3.5/5
Kde je přesně hranice vůči geodesic latent-space metodám

Arvanitidis et al. jsou potenciálně nejnebezpečnější prior art, pokud by novelty claim zněl příliš široce. U nich je obecná distance

[ d(x,y)

\inf_{\gamma(0)=x,\gamma(1)=y} \int_0^1 |\dot\gamma(t)|_{G(\gamma(t))},dt, ]

tedy vzdálenost doslova path length. 
4

U vás je však zásadně jiný objekt:

[ \gamma_{s,g}(t)

\text{hidden state generated by learned inference dynamics}, ]

a neřešíte v inference explicitně

[ \inf_\gamma L(\gamma). ]

Model je supervised pouze výslednou BFS distance a sám si musí vytvořit computational trajectory, jejíž délka tuto vzdálenost kóduje. To je přesně místo, kde podle mé rešerše geodesic literature končí a vaše myšlenka začíná.

Velmi podobné rozlišení platí pro neural ODE. Finlay et al. mají continuous latent trajectory a integrál lokální kinetic quantity, ale integral funguje jako regularizer požadující „levnější“ dynamics; není to hlavní supervised prediction. 
34

Recentní Truth as a Trajectory je důležitá terminologická pojistka: protože používá přesně diskrétní

[ \sum_\ell|h_{\ell+1}-h_\ell|, ]

nedoporučoval bych tvrzení „we introduce measuring a network’s hidden trajectory by summing layer-to-layer displacements“. Tento matematický diagnostic už existuje. Bezprostředně obhajitelnější je:

we use this trajectory length as the model’s task output and train it to equal a goal-conditioned geodesic/BFS distance.

Právě tento přechod od diagnostic/regularizer k value parameterization jsem v nalezených pracích nenašel. 
37

Eikonal metody jsou taskově blízko, mechanicky daleko

Eikonal distance field je řešením lokální PDE typu

[ |\nabla D(x)|_{G^{-1}}\approx 1, ]

s boundary condition (D(g)=0). Neural Eikonal Solver proto dokáže po naučení přímo odpovídat na distance queries a z gradientu rekonstruovat globally minimizing geodesics. To je velmi blízko vašemu pure distance oracle use-case, ale síť stále reprezentuje scalar field (D); nedává distance význam „celkové délky vlastní inference dynamiky“. 
39

Front propagation/Fast Marching a differentiable value iteration zároveň představují klasický mechanismus, kde se distance šíří lokálně a každý environment edge přidává lokální cost. Avšak počet propagation iterations je spojen s tím, jak daleko se informace musí dostat, nebo se nakonec čte arrival-time/value field. Ani tento směr tedy nevytváří váš fixed-compute property. VIN a neural Bellman–Ford jsou nejbližší neural versions tohoto principu. 
26

Cognitive maps, successor representation a mentální simulace

Cognitive/neuroscience literatura poskytuje velmi dobré konceptuální analogy, ale nic, co by se podle nalezených modelů blížilo exact computational readoutu.

Práce	Mechanismus	Sčítá latentní movement do distance estimate?	Goal/replay vztah	Blízkost
Stachenfeld, Botvinick & Gershman, “The Hippocampus as a Predictive Map”, Nature Neuroscience 2017	Hippokampální reprezentace je interpretována skrze successor representation: state kóduje discounted future-state occupancies, což vytváří predictive cognitive map. 
43
	Ne	SR zachycuje reachability a prostorovou strukturu	2.5/5
Momennejad, Russek, Cheong, Botvinick, Daw & Gershman, “The Successor Representation in Human Reinforcement Learning”, Nature Human Behaviour 2017 
44
	Behaviorální data podporují predictive/SR-like representation budoucích states.	Ne	Goal/value lze recombinovat s cached transition predictions	2/5
Mattar & Daw, “Prioritized Memory Access Explains Planning and Hippocampal Replay”, Nature Neuroscience 2018 
45
	Replay je interpretován jako prioritizované použití zkušeností pro value backups/planning.	Ne; Bellman/value updates nejsou normy pohybu neural representation	Silně goal/value relevantní	2.5/5
Jensen, Hennequin & Mattar, “A Recurrent Network Model of Planning Explains Hippocampal Replay and Human Behavior”, Nature Neuroscience 2024 
46
	Recurrent neural agent meta-learns, kdy interně plánovat; rychlé network dynamics generují replay-like internal sequences.	Ne	Ano, velmi blízký „internal recurrent simulation supports goal-directed decisions“	3/5

Empirická hippocampal literature navíc ukazuje prospective/replay sequences reprezentující možné či budoucí trajektorie k cílům a spojuje hippocampus s plánováním. Replay je však sekvence reprezentovaných environment states, nikoli obecně abstraktní hidden-computation trajectory, jejíž Euclidean displacement se integruje jako subjektivní distance estimate. 
47

Pro váš paper bych tedy neuroscience framing používal opatrně: lze motivovat ideu „distance by internal simulation / internal trajectory“, ale nenašel jsem zde precedent pro přesnou identitu

[ \text{estimated distance}

\text{arc length of internal recurrent state trajectory}. ]

Jensen et al. jsou nejzajímavější proto, že explicitně používají rychlou recurrent network dynamics jako mechanismus planning/replay, ale jejich model není distance oracle a délka či rychlost této neural dynamics se nečte jako shortest-path value. 
48

SR literatura nabízí jinou, téměř opačnou možnost: místo simulování jednoho latentního flow předpočítává discounted predictive occupancy. Proto může generovat cognitive-map geometry bez toho, aby při každém query něco „odintegrovala“ podél hidden trajectory. 
44

Syntéza podobnosti a tři nejbližší předchůdci

Následující matice shrnuje komponenty, které jsou pro váš novelty claim podle mého názoru nejdůležitější.

Legenda: ✓ přímý precedent, ~ částečný/analogický, ✗ chybí.

Kandidát	Iterativní flow	Increment-sum / arc-length readout	Goal re-injection / recall	Fixed-(T) budget-free distance extrapolation	Distance-only supervision
Vaše metoda	✓	✓	✓	✓	✓
Bansal et al. 2022, Recall recurrent reasoning 
1
	✓	✗	✓	✗ – harder ⇒ more iterations	✗
Arvanitidis et al. 2018, latent geodesics 
4
	~ – explicit optimized curve	✓ matematicky	✗	~ – distance magnitude není počet NN layers, ale geodesic solving má vlastní compute	~ – geometrická distance, ne BFS supervision
Hidden-trajectory arc length: Hasani et al. 2021 / 2026 diagnostic work 
3
	✓	✓ jako měřená statistika	✗	✗	✗
VIN / neural Bellman–Ford 
26
	✓	~ – local path cost je v Bellman algebra, ne (|\Delta h|)	~ goal boundary	✗ – propagation depth matters	~ shortest-path supervision/algorithm traces
QRL / IQE / temporal quasimetric 
49
	✗	✗	goal provided to endpoint readout	~ structural/combinatorial generalization	✓/~
Search on Replay Buffer 
15
	✗ uvnitř predictor	~ sčítá local distances po environment path	goal query	~ long routes stitched from local edges	~
Neural Eikonal Solver 
50
	✗ při scalar inference	✗ – PDE zaručuje field, path length je underlying geometry	✓/~ boundary/start conditioning	✓/~ arbitrary-distance queries po naučení fieldu	✗ – physics-informed, bez distance labels
Nejbližší práce celkově

Bansal et al., NeurIPS 2022 — closeness 4/5. Anticipují weight sharing across thinking steps, recurrent iterative flow a především recall/re-injection původní query při každém kroku; navíc explicitně cílí na easy-to-hard extrapolation. Neanticipují arc-length readout, goal-conditioned distance target ani fixed-compute extrapolation. Jejich generalization principle je „harder ⇒ think longer“, zatímco váš je „farther ⇒ move farther in latent space during the same number of thoughts“. 
51

Arvanitidis, Hansen & Hauberg, ICLR 2018 — closeness 4/5. Anticipují nejdůležitější matematický krok: distance is the integral of local latent speed along a path. Neanticipují, že ta path je inference trajectory weight-shared network; neexistuje recall, BFS supervision ani pure learned goal-conditioned value oracle. Geodesic je řešena jako shortest curve v předem vzniklé latent geometry. 
2

Hidden-trajectory arc-length literature, zejména Hasani et al. AAAI 2021 a velmi recentní Truth as a Trajectory 2026 — mechanická closeness 4/5, tasková mnohem menší. Anticipují měření neural trajectory pomocí její arc length a recentní práce dokonce používá přesný diskrétní výraz (\sum_\ell|h_{\ell+1}-h_\ell|). Nikdo z těchto předchůdců ale podle nalezených zdrojů neučí tuto quantity jako vlastní supervised output, natož jako state-goal shortest-path value. 
3

Těsně za nimi bych dal VIN / Neural Execution of Bellman–Ford: taskově jsou blíž než arc-length papers, protože iterativně počítají shortest-path-like quantities, ale mechanický readout je podstatně vzdálenější. 
26

Verdikt o novosti
Co již literatura podle rešerše jednoznačně anticipuje

Iterativní weight-shared inference. Universal Transformers, recurrent deep-thinking networks, neural algorithmic reasoners, VIN/GPPN a DEQ mají tuto ideu ve více podobách. 
52

Recall / opakované přidávání původního vstupu. Bansal et al. jsou zde přímý precedent; jejich explicitním cílem je zabránit recurrent computation v zapomenutí zadání a používají původní problem instance jako reference pro každý další algorithmic step. 
1

Distance as a path integral. Riemannovská geometrie a latent-geodesic literature tuto interpretaci používají standardně; Neural ODE a optimal-transport práce navíc běžně integrují speed/kinetic quantities podél continuous trajectories. 
53

Arc length hidden activations. Jako diagnostic/complexity measure existuje v recurrent/continuous networks a k roku 2026 i přímo ve formě sumy Euclidean displacement norms přes successive hidden representations. 
3

Goal value as distance / quasimetric. QRL, PQE/IQE, MRN, Dynamical Distance Learning, contrastive temporal distances a Eikonal value learners jasně pokrývají tuto část. 
49

Local-to-distant goal generalization. Je explicitně předmětem moderní horizon-generalization a temporal-distance literatury a external-search approaches skládají dlouhé cesty z learned local quantities. 
54

Co jsem v literatuře nenašel spojeno

Nejdůležitější mezera je velmi specifická:

Žádná nalezená GCRL/value práce nepoužívá total latent motion vykonaný při interním recurrent inference jako samotnou hodnotu (V(s;g)).

Konkrétně jsem nenašel model, kde by simultaneously platilo:

[ V(s;g) \propto \sum_{t=0}^{T-1} \sum_i |z_i^{t+1}-z_i^t|, ]

(z^{t+1}) bylo vytvořeno shared recurrent blockem, (s/g) byly během této dynamiky opakovaně zpřístupněny, targetem byl pouze shortest-path distance a extrapolace na vzdálenější goals probíhala při stejném (T). Nejbližší literatury pokrývají vždy pouze jednu nebo dvě strany tohoto průsečíku. 
55

Stejně tak jsem nenašel GCRL práci, která by využívala princip:

distance magnitude is carried by latent velocity rather than computational duration.

To je podle mého soudu možná konceptuálně ještě výraznější novelty než samotná sum-of-norms formula. Recurrent extrapolation papers typicky reprezentují zvýšenou obtížnost zvýšeným počtem recurrences, zatímco u vás mohou stejné (T) kroky generovat libovolně větší total variation. 
56

Jak bych novelty claim formuloval v článku

Nejsilnější obhajitelná formulace by byla přibližně:

Prior goal-conditioned distance models read values directly from state-goal embeddings or scalar value heads, while iterative planners and recurrent reasoning networks read their predictions from the final recurrent state. In contrast, we parameterize the goal-conditioned distance as the discrete arc length of the model’s own recurrent latent inference trajectory, summing nonnegative hidden-state displacement magnitudes across a fixed number of weight-shared iterations.

První část je podpořena quasimetric, contrastive-value a recurrent-planning literaturou; druhá popisuje specifikum vaší metody. 
49

Následující tvrzení bych naopak nepoužíval bez kvalifikace:

“We are the first to represent distance as a latent path length.”

To by bylo příliš široké vzhledem k Riemannian latent-space práci. 
4

Stejně tak:

“We are the first to sum hidden-state displacements across network depth.”

je po recentních trajectory-geometry pracích obtížně obhajitelné. 
37

A:

“We introduce recall in iterative networks.”

je přímo v konfliktu s Bansal et al. 
57

Celkové posouzení

Exact-match prior art: nenalezen.

Partial anticipation: ano, poměrně silná, ale rozdělená mezi několik nespojených komunit.

Moje výsledné hodnocení jednotlivých komponent:

Komponenta vaší metody	Stav prior art
Goal-conditioned value interpretovaná jako distance	Dobře etablováno. QRL, DDL, temporal distances, Eikonal GCRL. 
49

Weight-shared iterative network	Dobře etablováno. Universal Transformer, deep thinking, neural algorithms. 
52

Re-injection start/goal při každém kroku	Silně anticipováno obecným Recall principem Bansal et al.; specifická tokenová start+goal implementace může být vaše. 
1

Path length jako (\int|\dot z|)	Klasický geometric precedent. 
4

(\sum_t|h_{t+1}-h_t|) pro neural hidden states	Existuje jako analysis/diagnostic quantity. 
37

Tato suma jako supervised network output/value	V prohledané literatuře nenalezeno.
Goal-conditioned shortest-path value = tato suma	V prohledané literatuře nenalezeno.
Fixed (T), ale extrapolace distance magnitude přes increment size	V prohledané literatuře nenalezeno; kontrastuje s „think longer“ extrapolation. 
56

Distance-only labels, žádná policy, unseen-environment extrapolation	Jednotlivé části mají precedenty, ale nenašel jsem je kombinované s arc-length inference readoutem. 
58

Proto bych pro paper formuloval závěr jako „nová parametrizace goal-conditioned distance/value“, nikoli jako „nový pojem distance“, „nový recurrent architecture“ nebo „nový latent path integral“.

Další komunity a search terms, které stojí za kontrolu před definitivním priority claimem

Největší residual risk podle mého názoru není v mainstream GCRL — tam je mechanická mezera poměrně jasná — ale ve vzdálenějších komunitách, kde by se podobný scalar mohl používat pod jiným názvem. Hledal bych zejména kombinace výrazů “activation trajectory arc length”, “representation path length”, “total variation across depth”, “latent kinetic distance”, “computational trajectory length”, “residual-flow length”, “neural path length as prediction”, “monotone additive recurrent output” a “anytime cumulative evidence network”. Recentní hidden-geometry práce ukazují, že terminologie „arc length of hidden trajectory“ skutečně existuje mimo RL. 
37

Další potenciální zdroje částečných precedentů jsou neural survival/temporal point processes, kde se scalar cumulative hazard získává integrací lokálně nezáporné intensity; optimal-control/action functional learning, kde value může být integrál running cost; learned numerical solvers a amortized geodesic solvers; neural A/differentiable shortest-path layers*; a energy-based iterative inference, kde se přes inference iterations sleduje pokles energie. Tyto směry mohou obsahovat obecný princip „output/objective = accumulated nonnegative local quantity“, ale rešerše výše nenašla nic, co by současně používalo normu pohybu vlastního hidden state jako goal-conditioned shortest-path prediction. Neural A*/Q*-style search například stále používá environment transition costs a heuristics, nikoli latent displacement readout. 
59

Nejdůležitější bibliografické trojici pro positioning bych tedy postavil takto: Bansal et al. 2022 pro recurrent recall a extrapolaci, Arvanitidis et al. 2018 pro distance-as-latent-path-integral a Wang et al. 2023 / Myers et al. 2024 pro goal-conditioned value-as-distance. Vaše novost leží přesně v průsečíku těchto tří linií, který žádná z nich sama neobsahuje. 
57
