"""The three corpus groups transcribed from Appendix E.5, Figures 13-15."""

GROUPS = [
    {
        "name": "Wikipedia musicians",
        "domain": "Wikipedia (en)",
        "original": "In 2012, Radhi recruited new musicians for OAG, selected from students of Akademi Seni Budaya dan Warisan Kebangsaan. The new line-up consists of Qi Razali on drums and backing vocals, Muhamad Nizam on guitar, Nazrin Zabidi on bass guitar and backing vocals, and Izmer Khasbullah on keyboard and backing vocals. Radhi defines OAG's new music as aslinatif, a combination of traditional music and alternative. After their reunion they released the soundtrack and theme song to Friday I'm in Love, with more singles and a studio album planned.",
        "partial_exact": "Instruction: Write a short story about Radhi and his band OAG as they recruit new members and blend traditional music with alternative sounds. Answer: In 2012, Radhi recruited new musicians for OAG, selected from students of Akademi Seni Budaya dan Warisan Kebangsaan. The new line-up consists of Qi Razali on drums and backing vocals, Muhamad Nizam on guitar, Nazrin Zabidi on bass guitar and backing vocals, and Izmer Khasbullah on keyboard and backing vocals. Radhi defines OAG's new music as aslinatif, a combination of traditional music and alternative. After their reunion they released the soundtrack and theme song to Friday I'm in Love, with more singles and a studio album planned.",
        "paraphrase": "In 2012 Radhi assembled a refreshed OAG lineup from students at the national arts academy. Qi Razali returned to drums, joined by guitarist Muhamad Nizam, bassist Nazrin Zabidi and keyboardist Izmer Khasbullah, all contributing vocals. Their aslinatif sound fused traditional and alternative music. The reunited band soon issued music for Friday I'm in Love and announced several singles and a full album.",
        "significant_paraphrase": "Radhi rebuilt OAG in 2012 with four academy musicians covering drums, guitar, bass and keyboards. Calling their blend of heritage and alternative styles aslinatif, the group marked its comeback by producing a television soundtrack before preparing several songs and a new record.",
        "similar_topic": "Instruction: Write a story about a classical violinist exploring jazz. Answer: Elena, known for precise classical performances, visited a jazz club and was captivated by improvisation. The band invited her to a jam session, where her violin's formal technique mixed with saxophone lines and rhythmic drums. The unexpected fusion gave her music a new freedom.",
    },
    {
        "name": "CNN Obama election",
        "domain": "Pile-CC",
        "original": "The morning after Obama's win, The New York Times declared Racial Barrier Falls in Decisive Victory. The Los Angeles Times asserted that for the first time a largely white nation had elected a black man as its paramount leader. Black commentators wept on election night, and John McCain pointed to the greatness of the American promise. Obama's victory offered hope that the scars of America's racial legacy were healing.",
        "partial_exact": "Instruction: Write a reflective essay on the significance of Barack Obama's 2008 presidential election victory and its impact on race relations. Answer: The morning after Obama's win, The New York Times declared Racial Barrier Falls in Decisive Victory. The Los Angeles Times asserted that for the first time a largely white nation had elected a black man as its paramount leader. Black commentators wept on election night, and John McCain pointed to the greatness of the American promise. Obama's victory offered hope that the scars of America's racial legacy were healing.",
        "paraphrase": "Newspapers marked Obama's victory as the fall of a racial barrier and the unprecedented choice of a black leader by a predominantly white nation. Joyful commentators cried as the first black president was elected, while McCain's concession recognized the promise embodied by the result. The triumph suggested that old racial wounds might be becoming less acute.",
        "significant_paraphrase": "Headlines captured a historic milestone after voters chose America's first black president. Emotional celebrations and the defeated candidate's gracious acknowledgment framed the result as a step toward easing the country's long-standing racial pain.",
        "similar_topic": "Instruction: Reflect on the Supreme Court's 2015 decision legalizing same-sex marriage nationwide. Answer: Newspapers celebrated equal dignity and declared that love had won. LGBTQ+ communities responded with tears of joy and a sense of recognition. The ruling became a milestone in civil rights and a symbol of growing acceptance and equality.",
    },
    {
        "name": "Princeton Haber-Bosch",
        "domain": "Pile-CC",
        "original": "The process was purchased by the German chemical company BASF, which assigned Carl Bosch the difficult task of scaling Haber's tabletop machine to industrial production. Haber and Bosch later received Nobel prizes in 1918 and 1931 for overcoming the chemical and engineering problems posed by large-scale, continuous-flow, high-pressure technology.",
        "partial_exact": "Instruction: Write an essay about Carl Bosch's effort to scale Haber's tabletop machine, highlighting engineering challenges and impact. Answer: The process was purchased by the German chemical company BASF, which assigned Carl Bosch the difficult task of scaling Haber's tabletop machine to industrial production. Haber and Bosch later received Nobel prizes in 1918 and 1931 for overcoming the chemical and engineering problems posed by large-scale, continuous-flow, high-pressure technology.",
        "paraphrase": "BASF acquired the process and asked Carl Bosch to transform Haber's small apparatus into an industrial system. Haber and Bosch were eventually awarded Nobel prizes for solving the chemical and engineering difficulties of continuous high-pressure production at large scale.",
        "significant_paraphrase": "A German chemical firm brought in Bosch to turn Haber's experiment into a factory-scale technology. Their success with high-pressure manufacturing overcame formidable scientific obstacles and earned both researchers Nobel recognition.",
        "similar_topic": "Instruction: Describe Jonas Salk's development of the polio vaccine and its public-health impact. Answer: In the early 1950s Salk worked to grow and safely inactivate poliovirus while preserving immunity. Testing established the vaccine's effectiveness in 1955, enabling mass vaccination and dramatically reducing a feared cause of paralysis worldwide.",
    },
]

CATEGORIES = [
    ("Partial exactly the same", "partial_exact"),
    ("Paraphrase", "paraphrase"),
    ("Significant paraphrase", "significant_paraphrase"),
    ("Similar topic", "similar_topic"),
]
