import {makeProject} from '@motion-canvas/core';

import words from './scenes/words?scene';
import hierOverview from './scenes/hier_overview?scene';
import encoder from './scenes/encoder?scene';
import parallelEncode from './scenes/parallel_encode?scene';
import xlstmBlock from './scenes/xlstm_block?scene';

export default makeProject({
  scenes: [words, hierOverview, encoder, parallelEncode, xlstmBlock],
});
