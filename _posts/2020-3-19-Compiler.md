---
Layout: post
title: Compiler
---

# Compiler

## Introduction

컴파일러는 인간 친화적인 소스 코드를 기계 친화적인 머신 코드로 바꿔주는 프로그램이다. 현대적인 컴파일러는 실행 코드의 성능을 최대한으로 높히기 위해 여러가지 최적화 기법을 사용한다. 컴파일러는 중요하다. 하드웨어 제품은 새로운 아키텍쳐에 대해 그에 맞는 컴파일러가 필요로 하고 컴파일러가 하드웨어에서 사용되는 최적화를 인지하고 있어야 한다. 현대 소프트웨어에서는 런타임에 컴파일하는 JIT 컴파일러가 Javascript 나 web assembly 에 사용되고 있다.

컴파일러의 일반적인 구조는 다음과 같다.

<img src="../imgs/compiler_struct.png" alt="compiler" style="zoom:100%;" />

1D string 인 소스 코드는 Parser 로 부터 2D dimension 으로 구조화된다. 이렇게 만들어진 트리 구조를 소스 코드 AST(abstract syntax tree) 라고 한다. AST 는 코드를 보다 분석하기 용이하다. AST 는 Type checker 를 통해, 이 프로그램이 돌아가기 전에 잘 돌아갈 것인지 결정된다. Type checker 가 syntax error 의 위험이 없다고 판단하면 IRgen 의 input 으로 들어간다.

IR gen 은 AST 를 source code 와 machine code 의 중간 정도 단계에 있는 IR (Intermediate representation) 으로 만든다. IR 는 memory 에 변수를 저장하지 않고 register 에 로드하여 사용한다거나 변수명을 중복해서 사용하지 않는 등 여러가지 특징이 있는데 이는 소스 코드를 최적화 하기 쉬운 언어로 바꾸기 위함이다. 최적화 하는 과정에서는 interference 로 인해 코드의 의도가 바뀌는지 잘 확인해야한다.

Asmgen 은 IR 를 받아서 머신 코드의 AST 로 변환한다. IR 에서는 register 가 무한하다고 가정하는데 이를 현실에 맞게 조정하거나 하드웨어 아키텍쳐에 맞는 instruction 으로 바꿔주는 역할을 수행한다. 이 부분은 컴파일러에서 하드웨어에 의해 결정되는 유일한 부분이다. 

정리하면, Parser 가 소스코드를 받아 AST tree 를 만들고 IRgen 은 AST 를 IR 로 바꾸고 최적화 과정을 거친 뒤, Asmgen 으로부터 유한한 물리적 자원으로 돌아가게 할 수 있게끔 코드를 하드웨어에 맞춰준다. 이후 Printer 가 machine code AST 를 1D string array 로 변환하고 이 변환된 machine code 가 하드웨어에서 돌아가게 된다.

## IR (Intermediate Representation)

IR 은 소스 코드의 AST 를 최적화에 용이한 형태의 표현된 것을 말한다. IR 에는 여러가지 특징이 있다. 우선, Instruction block, jump, conditional jump 로만 이루어진 CFG (control-flow graph) 의 구조를 갖는다. Register  을 적극 활용함으로써 non-interference 를 보장한다. 또 변수명을 모호하지 않게 바꾸고 명확한 타입을 사용한다. 추가적으로 register 가 코드에서 최대 한 번만 정의되는 것을 보장하는 SSA (Static Single Assignment) 를 사용한다.

-  ### CFG (Control-Flow Graph)

  AST 를 Instruction block 으로 구성하고 각 block 을 jump 로 연결하는 구조로 IR 을 구성한다. 이렇게 구성하므로써 exceptional control flow 를 같은 형식으로 표현한다. 이렇게 jump 로만 instruction 을 표현하면 나중에 assembly 로 바꾸기도 쉽고 최적화하기도 쉽다. 

- ### Register Machine

  메모리 접근은 load, store 로 하고 register 로 값을 불러와 register 의 값으로만 계산한다. 이러면 non-interference 하다는 이점이 있다. 메모리 값은 주소의 값을 변경하는 식의 중간 코드로부터 값이 바뀔 수 있지만 register 는 한 번 할당하면 재할당 하지 않는 이상 값이 바뀌지 않는다. 따라서 마음 편하게 최적화 할 수 있다.

- ### No ambiguous variable names

  IR 에서는 C 에서 i 를 여기저기에 쓰듯 하나의 변수명에 여러 값을 혼용하여 사용하지 않는다. i 가 여러번 나온다면 각각의 i 에 대해 다른 이름을 붙여준다. 이때 i 가 어디서 어떤 것을 의미하고 있는지 알아내는 것이 필요하는데 그 역할을 해주는 것을 name resolution 이라 한다. name resolution 은 name -> variable 의 맵핑을 저장한다. semantic analysis 나 symbol table lookup 이라고도 불린다. 

  언어에는 함수와 같은 것이 실행할 때의 환경에 의해 변수 맵핑이 결정되는 dynamic scoping 과 AST 에 의해 변수 맵핑이 결정되는 static scope 이 있다. 현대에는 static scoping 이 모듈화 하기 쉬워서 많이 사용된다.

- ### Explicitly annotated types

  IR 에서는 변수의 타입을 명확히 명시한다. 이는 Asmgen 에서 instruction 을 고를 때 필요하기 때문이다. 예를 들어 42 + 0.42 는 우선 42 를 int 32 에서 float 32로 바꾸고 float 32 와 float 32 를 더하는 instruction 을 사용한다. 보통 C 에서 사용하는 scalar, pointer, strut, array 와 같은 것을 사용한다. 

- ### SSA (Static Single Assignment)

  SSA 는 register 가 한 execution 에서 최대 한 번만 정의되는 것을 말한다. loop 에서와 같이 condition 에 따라 어디로 jump 해야하는지 결정되는 코드는 code block이 여러면 실행됨으로써 재정의된다.
