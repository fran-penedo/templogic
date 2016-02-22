#!/bin/bash

sed -e 's/{\[/\[/g' \ # {[ to [
    -e 's/)}\\/\]/g' \ # )} to ]
    -e 's/\\wedge/\^/g' \ # \wedge to ^
    -e 's/\\vee/v/g' \ # \vee to v
    -e 's/\\leq/<=/g' \ # \leq to <=
    -e 's/y/x_0/g' \ # y to x_0
    -e 's/u/x_1/g' \ # u to x_1
    -e 's/\]/\] (/g' \ # Adds a ( after ], so that expressions are enclosed in parenthesis
    -e 's/(\([GF]\)/\1/g' \ # Adds a ( before G or F, to compensate the one above
    $1
