TEST=runs/baseline/test
DIR=runs/baseline/
PARAMS=runs/baseline/eval_params
echo 'Saving results to '$DIR'...'

cat $TEST | cut -f2 | sed 's/<TOK>/( <TOK>)/g' | sed 's/<EOS>.*//g' > $DIR/eval_gold
cat $TEST | cut -f3 | sed 's/<TOK>/( <TOK>)/g' | sed 's/<EOS>.*//g' > $DIR/eval_pred

echo 'Running EVALB...'
./EVALB/evalb -p $PARAMS $DIR/eval_gold $DIR/eval_pred
