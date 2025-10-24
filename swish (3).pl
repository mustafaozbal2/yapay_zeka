% Belirti varsa, 1 (evet), yoksa 0 (hayır)

% Ateşi değerlendir (38 ve üstü ise var sayılır)
belirti_var(ates, VarMi) :-
    write('Ateşin var mı? (evet/hayir): '), read(Cevap),
    (
        Cevap = evet ->
            write('Kaç derece?: '), read(Derece),
            (Derece >= 38 -> VarMi = 1 ; VarMi = 0)
        ;
        VarMi = 0
    ).

% Öksürük
belirti_var(oksuruk, VarMi) :-
    write('Öksürüğün var mı? (evet/hayir): '), read(Cevap),
    (Cevap = evet -> VarMi = 1 ; VarMi = 0).

% Nefes darlığı
belirti_var(nefes_darligi, VarMi) :-
    write('Nefes darlığın var mı? (evet/hayir): '), read(Cevap),
    (Cevap = evet -> VarMi = 1 ; VarMi = 0).

% Boğaz ağrısı
belirti_var(bogaz_agrisi, VarMi) :-
    write('Boğaz ağrın var mı? (evet/hayir): '), read(Cevap),
    (Cevap = evet -> VarMi = 1 ; VarMi = 0).

% Tat veya koku kaybı
belirti_var(tat_kaybi, VarMi) :-
    write('Tat veya koku kaybın var mı? (evet/hayir): '), read(Cevap),
    (Cevap = evet -> VarMi = 1 ; VarMi = 0).

% Ana kontrol (uzman sistemi)
korona_tahmini :-
    belirti_var(ates, A),
    belirti_var(oksuruk, B),
    belirti_var(nefes_darligi, C),
    belirti_var(bogaz_agrisi, D),
    belirti_var(tat_kaybi, E),

    Toplam is A + B + C + D + E,

    (
        Toplam >= 3 ->
            write('Korona olabilirsin! Bir sağlık kuruluşuna başvur. 🏥')
        ;
            write('Belirtiler az, büyük ihtimalle korona değilsin. 🙂')
    ).
%korona_tahmini.
