"use client";

import SubjectPracticePage from "@/app/practice/[subject]/page";

// Next.js App Router already passes params to page components. Re-export the component
// to keep a single implementation while satisfying the /subjects/[subject] route.
export default SubjectPracticePage;
