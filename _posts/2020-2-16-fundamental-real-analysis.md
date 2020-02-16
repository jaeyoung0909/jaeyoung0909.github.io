---
Layout: post
title: Fundamental Real Analysis
use_math: true
---

## 실수

실수란 ordered field 중 least upper bound property 를 가지고 있는 것이다. 이 말에 포함된 단어들을 하나씩 분석하겠다.

ordered set 는 어떠한 집합에서 임의의 두 원소 사이에 <, >, = 중 하나의 관계를 가지고 있는 것이다.

field 란 어떤 집합에서 덧셈과 곱셈이 정의되고 그것들이 몇 가지 특성을 만족하는- associative, commutative 등 10가지- 집합이다.

ordered field 는 어떠한 집합이 ordered set 이고 field 이면서 field 에서 정의된 덧셈과 곱셈, order 의 관계가 다음 두 가지 조건을 만족하는 것이다.
$$
1.\ x + y < x + z if x, y, z \in F\ and\ y < z  \\
2.\ xy > 0 if x \in F, y \in F, x > 0, and\ y > 0
$$
least upper bound property 는 ordered set 에서 bounded above 한 subset 은 항상 least upper bound (supremum) 를 가지고 있는 특성이다.

이러한 실수의 특성으로 인해 archimedean property, Q 의 density in R, 1-1 correspondence of function $x^{1/n}$ 을 보일 수 있다.

 $thm$ archimedean property : $\exist n \ s.t. \ nx > y \ for \ y \in R, \ positive \ x \in R$

$pf$ 

suppose there is no exist such n and let A be the set of nx where n is positive integer. then nx is upper bounded by y, so there exist sup A by least upper bound property. sup A - x < sup A and sup A - x is not least upper bound, there exist m s.t mx > supA - x by definition of supremum. then (m-1)x > sup A so (m-1)x is upper bound of A but it is in A. => contradict!

$thm$ Q is dense in R, which implies that there exist p $\in$ Q s.t. x<p<y if x, y $\in$ R and x < y 

$pf$

there exist n s.t. (y-x)n > 1 by archimedean.

there exist m s.t. m-1 $\le$ nx $\lt$ m by archimedean.

so p is $m \over n$    

$thm$ $\exist ! y \ s.t. \ y^n=x \ for \ x\in R, integer \ n $

$pf$ 

E  를 $t^n < x$ 인 t 들의 집합이라고 정의하자. 이 집합은 실수 집합이고 upper bound 가 있으니까 sup 이 있다. 이 sup이 n 제곱 했을 때 $x$ 보다 크지도 않고 작지도 않음을 보이면 같음을 보일 수 있다. 과정은 너무 테크닉적이라 생략한다. 

**실수는 존재한다.**

$pf$ 다음과 같이 cuts 을 정의하자. 

cuts 는 Q 의  proper subset 이다. 이것은 not empty 이다. $p \in cuts$ 에 대해, $p < r\ for\ some\ r \in \alpha \in cuts $ and if q < p for $q \in Q$, $ q\in cuts$ 이다. 그렇다면 cuts 은 least upper bound property 을 가진 ordered field 이고 따라서 R 이다. 더 나아가, cuts 에서의 subfield 을 Q 와 isomorphic 하게 정의할 수 있다. 또, 이러한 R 은 유일하다. 즉, 만약 어떤 집합이 실수의 정의를 따른다면 그 집합은 실수와 isomorphic 하고 Q 와 isomorphic 한 집합을 subfield 로 갖는다.



앞으로 실수에 관하여 논의할 것인데, 그에 앞서 기본적으로 알아야 할 위상 개념을 정리하겠다.

## Basit Topology 

$def$

