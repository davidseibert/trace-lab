/**
 * Word lists shared by BOTH morphology lenses (merge and split) — identical on
 * purpose, so the two algorithms can be compared on the same data. They
 * approach the same objective from opposite extremes.
 *
 * "word n" sets a corpus frequency; bare words default to 1. Frequency drives
 * the merges/splits: lists are chosen so shared stems + affixes pay off.
 */
export const MORPH_SAMPLES: Record<string, string> = {
  'verb inflection': `walk 6
walks 3
walking 8
walked 4
talk 5
talks 2
talking 7
talked 3
jump 2
jumps 1
jumping 3
jumped 2`,
  'un- prefix': `happy 5
unhappy 3
kind 4
unkind 2
clear 4
unclear 2
lock 3
unlock 4
fair 3
unfair 2`,
  plurals: `cat 6
cats 4
dog 7
dogs 5
bird 3
birds 2
hand 4
hands 3
book 5
books 4`,
  'agreement (Spanish-ish)': `gato 4
gatos 3
gata 2
gatas 1
perro 5
perros 4
perra 2
perras 1
nino 3
ninos 2
nina 2
ninas 1`,
  'frequency matters': `the 100
then 4
there 5
they 8
them 6
a 80
at 7
an 9`
};

export const MORPH_DEFAULT_SAMPLE = 'verb inflection';
