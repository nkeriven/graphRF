% Revision: 02-01-2018
% Revision History
%   02-01-2018 : Follow WSU PhD/Masters format requirements as of 02-01-2018
%   07-10-2008 : Corrected Alignment of signature boxes on Masters/PhD Approval page
%   07-25-2007 : Corrected some spelling errors
%   05-16-2006 : Added etd option and moved most packages from class file to template
%   05-15-2006 : Initial version.
%
% Known bugs
%   Having a tiny bit of the abstract spill to second page defeats page number removal.
%   Workaround: make the abstract a little longer or a little shorter.
%
% You can supply
% the following optional arguments in the square brackets to
% specify the thesis type:
%
%   senior  : Produces the senior thesis preliminary pages (default)
%   honors  : Produces the honors thesis preliminary pages
%   masters : Produces the masters thesis preliminary pages
%   phd     : Produces the PhD dissertation preliminary pages
%
% The default format is appropriate for printing, with blank pages
% inserted after the preliminary pages in twoside mode so you can
% send it directly to a two-sided printer. However, for ETD
% submission the blank pages need to be removed from the final output.
% The following option does this:
%
%   etd     : Produces an electronic copy with no blank pages in the preliminary section
%
% The rest of the class options are the same as the regular book class.
% A few to remember:
%
%   oneside : Produces single sided print layout (recommended for theses less than 50 pages)
%   twoside : Produces single sided print layout (the default if you remove oneside)
%
% The BYUPhys class provides the following macros:
%
%   \makepreliminarypages : Makes the preliminary pages
%   \clearemptydoublepage : same as \cleardoublepage but doesn't put page numbers
%                           on blank intervening pages
%   \singlespace          : switch to single spaced lines
%   \doublespace          : switch to double spaced lines
%
% ------------------------------------------------------------------------------------------------------
%
\NeedsTeXFormat{LaTeX2e} \ProvidesClass{WSUclass}

% ---------------------------- declarations -------------------------
%
% These macros are used to declare arguments needed for the
% construction of the preliminary pages

% Fix?
\newcommand{\cedp}{\newpage{\pagestyle{empty}
\cleardoublepage}}

% The year and month the degree is awarded
\newcommand{\Year}[1]{\gdef\@Year{#1}}
\newcommand{\Month}[1]{\gdef\@Month{#1}}

% The full name of the degree
\newcommand{\degree}[1]{\gdef\@degree{#1}}

% The full name of the degree
\newcommand{\DegreeShort}[1]{\gdef\@DegreeShort{#1}}

% The name of this document (thesis/dissertation)
\newcommand{\docname}[1]{\gdef\@docname{#1}}

% First line of title
\newcommand{\TitleTop}[1]{\gdef\@TitleTop{\mbox{\uppercase{#1}}}}

%Middle line of title
\newcommand{\TitleMiddle}[1]{\gdef\@TitleMiddle{\mbox{\uppercase{#1}}}}

% Second line of title
\newcommand{\TitleBottom}[1]{\gdef\@TitleBottom{\mbox{\uppercase{#1}}}}

% Acknowledgments text
\newcommand{\Acknowledgments}[1]{\gdef\@Acknowledgments{#1}}

% Abstract text
\newcommand{\Abstract}[1]{\gdef\@Abstract{#1}}

% Dedication text
\newcommand{\Dedication}[1]{\gdef\@Dedication{#1}}

% The author's name
\newcommand{\Author}[1]{\gdef\@Author{#1}}

% The institution name
\newcommand{\University}[1]{\gdef\@University{#1}}

% The department name
\newcommand{\Department}[1]{\gdef\@Department{#1}}

% The name of the advisor
\newcommand{\Advisor}[1]{\gdef\@Advisor{#1}}

% The title of the advisor
\newcommand{\AdvisorTitle}[1]{\gdef\@AdvisorTitle{#1}}

% The name of the committee member 2
\newcommand{\MemberA}[1]{\gdef\@MemberA{#1}}

% The tile of the committee member 2
\newcommand{\MemberATitle}[1]{\gdef\@MemberATitle{#1}}

% The name of the committee member 3
\newcommand{\MemberB}[1]{\gdef\@MemberB{#1}}

% The tile of the committee member 3
\newcommand{\MemberBTitle}[1]{\gdef\@MemberBTitle{#1}}

% The name of the committee member 4
\newcommand{\MemberC}[1]{\gdef\@MemberC{#1}}

% The tile of the committee member 4
\newcommand{\MemberCTitle}[1]{\gdef\@MemberCTitle{#1}}
% If you only have three members including your advisor then delete Member C (needs multiple deletes below too)

% The name of the department chair
\newcommand{\DepRep}[1]{\gdef\@DepRep{#1}}

% The title of the department chair (allow for associate chair, etc.)
\newcommand{\DepRepTitle}[1]{\gdef\@DepRepTitle{#1}}

% The name of the department undergraduate coordinator
\newcommand{\UgradCoord}[1]{\gdef\@UgradCoord{#1}}

% The name of the dean
\newcommand{\Dean}[1]{\gdef\@Dean{#1}}

% The title of the dean
\newcommand{\DeanTitle}[1]{\gdef\@DeanTitle{#1}}

% The name of the honors dean
\newcommand{\HonorsDean}[1]{\gdef\@HonorsDean{#1}}

% Set default values for fields
  \Year{1905}
  \Month{January}
  \Author{Author}
  \University{University}
  \Department{Department}
  \TitleTop{First line of title}
  \TitleMiddle{ }
  \TitleBottom{ } % default is empty
  \Acknowledgments{Acknowledgment text goes here.}
  \Abstract{Abstract text goes here.}
  \degree{Bachelor of Science}
  \docname{senior thesis}
  \Advisor{Advisor}
  \AdvisorTitle{Ph.D.}
  \MemberA{Committee Member A}
  \MemberATitle{Ph.D.}
  \MemberB{Committee Member B}
  \MemberBTitle{Ph.D.}
  \MemberB{Committee Member C}
  \MemberCTitle{Ph.D.}
  \DepRep{Department Chair Name}
  \DepRepTitle{chair}
  \Dean{Dean Name}
  \DeanTitle{Associate Dean}
  \HonorsDean{Honors Dean Name}
  \UgradCoord{Department Ugrad Coordinator }

% ---------------------------- options ------------------------------

% A command to switch to single spaced lines
\newcommand{\singlespace}{\renewcommand{\baselinestretch}{1}\small\normalsize}

% A command to switch to double spaced lines
\newcommand{\doublespace}{\renewcommand{\baselinestretch}{1.66}\small\normalsize}

% A command pirated from chngpage.sty
\DeclareRobustCommand{\ch@ngetext}{%
  \setlength{\@colht}{\textheight}\setlength{\@colroom}{\textheight}%
  \setlength{\vsize}{\textheight}\setlength{\columnwidth}{\textwidth}%
  \if@twocolumn%
    \advance\columnwidth-\columnsep \divide\columnwidth\tw@%
    \@firstcolumntrue%
  \fi%
  \setlength{\hsize}{\columnwidth}%
  \setlength{\linewidth}{\hsize}%
}

% A command to make margins right for the initial single sided business.
\newcommand{\preliminarymargins}{%
    \addtolength{\textwidth}{0in}%
    \addtolength{\evensidemargin}{0in}%
    \ch@ngetext%
    }

% A command to fix the margins after the initial single sided business.
\newcommand{\fixmargins}{%
    \addtolength{\textwidth}{0in}
    \addtolength{\evensidemargin}{0in}
    \ch@ngetext%
}

% Define the preliminary section for a senior thesis.
% The senior option is essentially ignored since it is the default
  \newcommand{\makepreliminarypages}{
    \preliminarymargins
    \titlepage
    \copyrightpage
     \dedicationpage
    \seniorapprovalpage
    \acknowledgmentspage
    \abstractpage
    \fixmargins
    \renewcommand{\clearemptydoublepage}{\cle@remptydoublep@ge}
  }

% Define the honors thesis preliminary section if the 'honors' option is specified
\DeclareOption{honors}{
  \renewcommand{\makepreliminarypages}{
    \preliminarymargins
    \honorstitlepage
    \copyrightpage
    \seniorapprovalpage
    \acknowledgmentspage
    \abstractpage
    \fixmargins
    \renewcommand{\clearemptydoublepage}{\cle@remptydoublep@ge}
  }
}

% Changes to masters thesis preliminary section if the 'masters' option is specified
\DeclareOption{masters}{
  \degree{Master of Science}
  \docname{thesis}
  \renewcommand{\makepreliminarypages}{
    \preliminarymargins
    \titlepage
    \copyrightpage
     \dedicationpage
    \masterapprovalpage
    \acknowledgmentspage
    \abstractpage
    \fixmargins
    \renewcommand{\clearemptydoublepage}{\cle@remptydoublep@ge}
  }
}

% Changes to PhD preliminary section if the 'phd' option is specified
\DeclareOption{phd}{
  \degree{Doctor of Philosophy}
  \docname{dissertation}
  \renewcommand{\makepreliminarypages}{
    \preliminarymargins
    \titlepage
    \copyrightpage
    \phdapprovalpage
    \acknowledgmentspage
    \abstractpage
    \fixmargins
    \renewcommand{\clearemptydoublepage}{\cle@remptydoublep@ge}
    }
}

% --------------------- Some commands to handle the single sided preliminary pages ------------------

% Define the '\clearemptydoublepage' command to clear pages but not number any blank pages inserted.
% This is taken from the BYUThesis.cls file
\let\cle@rdoublep@ge\cleardoublepage
\newcommand{\cle@remptydoublep@ge}{
  \clearpage
  \if@twoside
  \ifodd\c@page\else
  \fi\fi
  {\pagestyle{empty}\cle@rdoublep@ge}}
\newcommand{\clearemptydoublepage}{\cle@remptydoublep@ge}


% Create an abstract environment which is single sided, even in a double sided book.
% again, this was taken from BYUThesis.cls
\def\skip@bstr@ctp@ges{\relax}
\def\@@skip@bstr@ctp@ges{%
  \if@twoside
   \ifodd\c@page\else
    \vbox{\vbox to \vsize{}}
    \clearpage\fi
   \else
  \fi
  \afterpage{\skip@bstr@ctp@ges}
}
\newenvironment{abstractenv}{
   \def\skip@bstr@ctp@ges{\@@skip@bstr@ctp@ges}
   \afterpage{\skip@bstr@ctp@ges \thispagestyle{empty}}
   \pagestyle{empty}
}

% Redefine above commands if etd option is specified.  The blank pages make printing nice,
% but they don't want them in the submitted PDF
\DeclareOption{etd}{
    \renewcommand{\clearemptydoublepage}{ \clearpage }
    \renewenvironment{abstractenv}{\afterpage{\pagestyle{empty}}\pagestyle{empty}}{}
  }

% ------------------------ Load the class and needed packages ---------------------------------

% Load the book class
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{book}}
\ProcessOptions\relax
\LoadClass[letterpaper,12pt]{book}

% The afterpage package is required to make single sided formal pages
% in a double sided environment
\RequirePackage{afterpage}

% Note: the hyperref package is required to make an appropriate ETD.
% However, we don't require it here since it is supposed to be the last
% package loaded and students may want to load other packages in the
% main tex file.  So that this class file doesn't crash if the student
% forgets to load hyperref, we have used the following commands below:
%
%   \providecommand\phantomsection{}
%   \providecommand\pdfbookmark[3][]{}
%
% These commands provide dummy versions of the macros, but won't
% bother the real versions if the hyperref package is loaded in the
% tex file.

% The tocloft package is required for formatting table of contents
\RequirePackage{tocloft}

% ---------------------------- main code ----------------------------

% If the \makepreliminarypages macro is not run, this never gets fixed.
  \setlength{\marginparwidth}{0pt}
  \setlength{\marginparsep}{0pt}
  \setlength{\oddsidemargin}{0in} %\setlength{\oddsidemargin}{0.23in}
  \setlength{\evensidemargin}{0in}
  \setlength{\textwidth}{6.5in} %\setlength{\textwidth}{6in}
  \setlength{\topmargin}{0in}
  \setlength{\headheight}{0in} %\setlength{\headheight}{0.125in}
  \setlength{\headsep}{0.025in}
  \setlength{\textheight}{8.875in} %\setlength{\textheight}{8.625in}
  \setlength{\footskip}{0.5in} %\setlength{\footskip}{0.625in}


% Format table of contents
\renewcommand{\cftpartfont}{\normalfont\bfseries} % \part font in ToC
\renewcommand{\cftchapfont}{\normalfont\bfseries} % \chapter font in ToC
\renewcommand{\cftchappagefont}{\normalfont}
\renewcommand{\cftpartpagefont}{\normalfont}
\renewcommand{\cftpartleader}{\cftdotfill{\cftdotsep}}
\renewcommand{\cftchapleader}{\cftdotfill{\cftdotsep}}
\renewcommand{\cftsecleader}{\cftdotfill{\cftdotsep}}
% \setcounter{tocdepth}{4}
% \setcounter{secnumdepth}{4}

\cftsetindents{chapter}{0.25in}{0.25in}
\cftsetindents{section}{0.5in}{0.35in}
\cftsetindents{subsection}{.85in}{0.5in}
\cftsetindents{subsubsection}{1.5in}{0.5in}

\setlength{\cftbeforepartskip}{.25em}
\setlength{\cftbeforechapskip}{.25em}
\setlength{\cftbeforesecskip}{-.5em}
\setlength{\cftbeforesubsecskip}{-.5em}

\addtocontents{toc}{~\hfill\textbf{Page}\par}

% Redefine the Table of Contents to deal with some blank page
% and bookmarking issues relating to ETD submission
% \let\TEMPtableofcontents\tableofcontents
% \renewcommand{\tableofcontents}{
%   \clearemptydoublepage
%   \providecommand\phantomsection{} \phantomsection
%   \addcontentsline{toc}{chapter}{Table of Contents}
%   \TEMPtableofcontents
% }


% Redefine the List of Figures to deal with some blank page
% and bookmarking issues
\let\TEMPlistoffigures\listoffigures
\renewcommand\listoffigures{
  \clearemptydoublepage  %%% <<< Removing causes page number problems
  \providecommand\phantomsection{} \phantomsection
  \addcontentsline{toc}{part}{LIST OF FIGURES}
  \setlength{\cftbeforefigskip}{-.5em}
  \renewcommand\cftfigafterpnum{\vskip1em\par}
  \TEMPlistoffigures
}
\let\mylistoffigures\listoffigures
% \let\TEMPlistoffigures\listoffigures
% \renewcommand{\listoffigures}{
%   \clearemptydoublepage  %%% <<< Removing causes page number problems
%   \providecommand\phantomsection{} \phantomsection
%   \addcontentsline{toc}{chapter}{LIST OF FIGURES}
%   \TEMPlistoffigures
% }


% Redefine the List of Tables to deal with some blank page
% and bookmarking issues
\let\TEMPlistoftables\listoftables
\renewcommand{\listoftables}{
  \clearemptydoublepage  %%% <<< Removing causes page number problems
  \providecommand\phantomsection{} \phantomsection
  \addcontentsline{toc}{part}{LIST OF TABLES}
  \setlength{\cftbeforetabskip}{-.5em}
  \renewcommand\cfttabafterpnum{\vskip1em\par}
  \TEMPlistoftables
}
\let\mylistoftables\listoftables
% \let\TEMPlistoftables\listoftables
% \renewcommand{\listoftables}{
%   \clearemptydoublepage  %%% <<< Removing causes page number problems
%   \providecommand\phantomsection{} \phantomsection
%   \addcontentsline{toc}{chapter}{LIST OF TABLES}
%   \TEMPlistoftables
% }


% Redefine the Bibliography to deal with a bookmarking issues
% \let\TEMPbibliography\bibliography
% \renewcommand{\bibliography}{
%   \providecommand\phantomsection{} \phantomsection
%   \addcontentsline{toc}{chapter}{REFERENCES}
%   \TEMPbibliography
% }

\renewcommand*{\contentsname}{TABLE OF CONTENTS}
\renewcommand*{\listfigurename}{LIST OF FIGURES}
\renewcommand*{\listtablename}{LIST OF TABLES}
\renewcommand*{\bibname}{REFERENCES}

\newcommand{\addchapheadtotoc}{
    \addtocontents{toc}{
        \protect\vspace{.5em}
        \protect\noindent \textbf{CHAPTER}\protect\par
        \protect%\vspace{-1em}
        }
    }

\newcommand{\addappheadtotoc}{
    \addtocontents{toc}{
        \protect\vspace{.5em}
        \protect\noindent \textbf{APPENDIX}\protect\par
        \protect%\vspace{-1em}
        }
    }


%---------------------------- The Preliminary Page Definitions --------------------------

% ============================== Title Page ===============================
\renewcommand{\titlepage}{
    \thispagestyle{empty}

    \includegraphics[scale=0.45]{figs/PERSYVAL.png}\hfill
    \includegraphics[scale=0.17]{ANRfinancé.jpg}
    \includegraphics[scale=0.17]{investissement-avenir.jpg}
         
          \par\noindent\rule{\textwidth}{0.4pt}
         
    \begin{center}
    \providecommand\pdfbookmark[3][]{} \pdfbookmark[0]{Title Page}{bm:Title}
    \vspace*{.5in}
    \@TitleTop\\[\baselineskip]
    \vspace*{.5in}
    \includegraphics[scale=0.3]{gipsa-logo.pdf}
    \includegraphics[scale=0.065]{logo_inp.png}
    \@TitleMiddle\\[\baselineskip]
    \@TitleBottom\\
    \vfill
    By: \\[\baselineskip] \MakeUppercase{\@Author} \\[\baselineskip]
    \vfill
    Supervised by: \\[\baselineskip]
    \MakeUppercase{Nicolas KERIVEN} \\[\baselineskip]
    \MakeUppercase{Nicolas Tremblay} 
    \vfill
    A \@docname~submitted for the degree of\\[\baselineskip]
    \MakeUppercase{\@degree} \\[\baselineskip]
    \MakeUppercase{\@University}\\
    \@Department\\[\baselineskip]
    \MakeUppercase{\@Month}~\@Year \\[\baselineskip]
   % \copyright\ Copyright by \MakeUppercase{\@Author},~\@Year\\
    %All Rights Reserved
        \footnote{This work has been partially supported by the LabEx PERSYVAL-Lab (ANR-11-LABX-0025-01) funded by the French program Investissement d’avenir}
    \end{center}

    
    \clearpage
    }

% ============================== Honors Title Page ========================
\newcommand{\honorstitlepage}{
    \thispagestyle{empty}
    \begin{center}
    \providecommand\pdfbookmark[3][]{} \pdfbookmark[0]{Title Page}{bm:Title}
    \vspace*{0.375in}
    \@TitleTop\\[\baselineskip]
    \@TitleMiddle\\
    \@TitleBottom\\
    \vfill
    by\\[\baselineskip]
    \@Author
    \vfill
    Submitted to Washington State University in partial fulfillment\\[\baselineskip]
    of graduation requirements for University Honors\\[2\baselineskip]
    School of Electrical Engineering and Computer Science\\[\baselineskip]
    \@Month~\@Year
    \vfill
    \end{center}
    \parbox[t]{2.75in}{
        Advisor: \@Advisor \\[.5\baselineskip]
        Signature: \hrulefill}
    \hfill
    \parbox[t]{2.75in}{
        Honors Dean: \@HonorsDean \\[.5\baselineskip]
        Signature: \hrulefill}
    \clearemptydoublepage
  }

% ======================== Copyright page ===============================
\newcommand{\copyrightpage}{
    \thispagestyle{empty}
    \providecommand\pdfbookmark[3][]{} \pdfbookmark[0]{Copyright}{bm:Copyright}
    \addtocounter{page}{-1}
    \vspace*{\fill}
    \vfill
    \begin{center}
    \copyright\ Copyright by \MakeUppercase{\@Author},~\@Year\\
    All Rights Reserved
    \end{center}
%    \vspace{1in}
    \clearpage
  }
\cedp

% =============================== Approval page =======================


\newcommand{\datebox}{
    \parbox[t]{1.5in}{
        \ \\[2\baselineskip]
        \rule{0.in}{0.4pt}\\

    }
}

\newcommand{\signaturebox}[1]{
    \parbox[t]{3.6in}{
        \ \\[2\baselineskip]
        \rule{3.6in}{0.4pt}\\
        #1
    }
}

\newcommand{\phdapprovalpage}{
    \thispagestyle{plain}
    \begin{center}
    \providecommand\pdfbookmark[3][]{} \pdfbookmark[0]{Graduate Committee Approval}{bm:ComAp}
    \vspace*{0.375in}
    \end{center}
    \noindent
    To the Faculty of Washington State University:
    \par\doublespace The members of the Committee appointed to examine the dissertation
of \MakeUppercase{\@Author} find it satisfactory and recommend that it be accepted.\\[\baselineskip]
    \par\singlespace
    \datebox\hfill\signaturebox{\@Advisor, \@AdvisorTitle, Chair}\\
    \datebox\hfill\signaturebox{\@MemberA, \@MemberATitle}\\
    \datebox\hfill\signaturebox{\@MemberB, \@MemberBTitle}\\
    \vfill
    \clearemptydoublepage
  }
\newcommand{\masterapprovalpage}{
    \iffalse
    \thispagestyle{plain}
    \begin{center}
    \providecommand\pdfbookmark[3][]{} \pdfbookmark[0]{Graduate Committee Approval}{bm:ComAp}
    \vspace*{0.375in}
    \end{center}
    \noindent
    To the Faculty of Washington State University:
    \par\doublespace The members of the Committee appointed to examine the thesis
of \MakeUppercase{\@Author} find it satisfactory and recommend that it be accepted.\\[\baselineskip]
    \par\singlespace
    \datebox\hfill\signaturebox{\@Advisor, \@AdvisorTitle, Chair}\\
    \datebox\hfill\signaturebox{\@MemberA, \@MemberATitle}\\
    \datebox\hfill\signaturebox{\@MemberB, \@MemberBTitle}\\
    \vfill
    \clearemptydoublepage
    \fi
  }

\newcommand{\seniorapprovalpage}{
    \thispagestyle{empty}
    \begin{center}
    \providecommand\pdfbookmark[3][]{} \pdfbookmark[0]{Department Approval}{bm:DepAp}
    \vspace*{0.375in}
    WASHINGTON STATE UNIVERSITY\\[3\baselineskip]
    DEPARTMENT APPROVAL\\[5\baselineskip]
    of a \@docname~submitted by\\[\baselineskip]
    \@Author\\[2\baselineskip]
    \end{center}
    \noindent
    This thesis has been reviewed by the research advisor,
    research coordinator, and department chair and has been
    found to be satisfactory.\\[\baselineskip]
    \datebox\hfill\signaturebox{\@Advisor, Advisor}\\
    \datebox\hfill\signaturebox{\@UgradCoord, Research Coordinator}\\
    \datebox\hfill\signaturebox{\@DepRep, \@DepRepTitle}\\
    \vfill
    \clearemptydoublepage
  }
% ======================= Acceptance Page ============================
\newcommand{\acceptancepage}{
    \thispagestyle{empty}%
    \begin{center}
    \providecommand\pdfbookmark[3][]{} \pdfbookmark[0]{Acceptance Page}{bm:Accept}
    \vspace*{0.375in}
    WASHINGTON STATE UNIVERSITY\\[3\baselineskip]
    \end{center}%
    \noindent%
    As chair of the candidate's graduate committee, I have read the
    \@docname\ of \@Author \ in its final form and have found
    that (1) its format, citations, and bibliographical style are
    consistent and acceptable and fulfill university and department
    style requirements; (2) its illustrative materials including
    figures, tables, and charts are in place; and (3) the final
    manuscript is satisfactory to the graduate committee
    and is ready for submission to the university library.\\[2\baselineskip]
    \datebox\hfill\signaturebox{\@Advisor\\Chair, Graduate Committee}
    \vskip 0pt plus 2fill
    \noindent Accepted for the Department\par\hfill%
    \signaturebox{\@DepRep, \@DepRepTitle\\School of Electrical Engineering and
    Computer Science}{} \vfill \noindent Accepted for the College\par\hfill
    \signaturebox{\@Dean, \@DeanTitle \\
    College of Sciences}
    \clearemptydoublepage
  }
  
% ======================= Dedication Page ============================
\newcommand{\dedicationpage}{
    \newpage
    \thispagestyle{plain}%
    \begin{center}
    \includegraphics[scale=0.69]{2020_presentation_Gipsa_ANR_ENG.pdf}
    %\includepdf{2020_presentation_Gipsa_ANR_ENG.pdf}
    \end{center}%
    \vfill
  }
\cedp

% ======================= Appendix Page ============================
\newcommand{\appendixpage}{
    \newpage
    %\thispagestyle{plain}%
    \thispagestyle{empty}
    \begin{center}
    \providecommand\pdfbookmark[3][]{} \pdfbookmark[0]{Appendix}{bm:APPENDIX}
    \vspace*{1.575in}
    \textbf{APPENDIX}\\[2\baselineskip]
    \end{center}%
    \vfill
    \clearpage
  }
\cedp

\let\TEMPappendix\appendix
\renewcommand{\appendix}{
  \addappheadtotoc
  \appendixpage
  \TEMPappendix
}
\let\myappendix\appendix

% ========================= Acknowledgments ==============================
\newcommand{\acknowledgmentspage}{
\newpage
    \providecommand\phantomsection{} \phantomsection
    \addcontentsline{toc}{part}{ACKNOWLEDGMENT}
    \thispagestyle{plain}
    \renewcommand{\baselinestretch}{1}\small\normalsize
    \begin{center}
    \providecommand\pdfbookmark[3][]{} \pdfbookmark[0]{Acknowledgments}{bm:Acknowledge}
    \vspace*{0.375in}
    \textbf{ACKNOWLEDGMENT}\\[3\baselineskip]
    \end{center}
    \renewcommand{\baselinestretch}{1.66} \small\normalsize%
    \@Acknowledgments
    \newpage
  }
\cedp

% ========================= Abstract ===================================

\newcommand{\abstractpage}{
    \thispagestyle{empty}
    \providecommand\phantomsection{} \phantomsection
    \addcontentsline{toc}{part}{ABSTRACT}
    \begin{abstractenv}
    \renewcommand{\baselinestretch}{1}\small\normalsize
    \begin{center}
    \providecommand\pdfbookmark[3][]{} \pdfbookmark[0]{Abstract}{bm:Abstract}
    \vspace*{.1in}
    \@TitleTop\\[\baselineskip]
    \@TitleMiddle\\[\baselineskip]
    \@TitleBottom\\[\baselineskip]
    Abstract\\[2\baselineskip]
    by \@Author, \@DegreeShort\\
    \@University\\
    \@Month~\@Year\\[3\baselineskip]
    \@DepRepTitle \@Advisor \\[2\baselineskip]
    \end{center}
    \renewcommand{\baselinestretch}{1.66}\small\normalsize
    \@Abstract
    \end{abstractenv}
    \clearpage
    }
\cedp


